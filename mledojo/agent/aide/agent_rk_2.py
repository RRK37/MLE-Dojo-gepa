import logging
import random
import humanize

import tiktoken
from typing import Any, Callable, cast, Dict, List, Tuple, Union
from dataclasses import dataclass

from mledojo.agent.aide.journal import Journal, Node, ExecutionResult
from mledojo.agent.aide.utils.config import Config
from mledojo.agent.aide.utils import data_preview
from mledojo.agent.aide.utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code
)
from mledojo.agent.aide.utils.util import (
    compile_prompt_to_md,
    opt_messages_to_list,
)
from mledojo.agent.aide.utils.metric import MetricValue, WorstMetricValue
from mledojo.chat import ChatClient, ModelSettings
from mledojo.agent.aide.pattern_analysis import (
    SolutionAnalyzer,
    PatternDiscovery,
    CorrelationAnalyzer,
    InsightGenerator,
    TemporalPatternAnalyzer
)

logger = logging.getLogger("aide")

ExecCallbackType = Callable[[str], Dict]

@dataclass
class LLMConfig:
    """Configuration for LLM API settings"""
    model_mode: str  # "local", "gpt", "gemini", "claude", etc.
    model_name: str  # Model name or engine name
    port: int = 8314  # Port for local model
    max_completion_tokens: int = 8192
    max_prompt_tokens: int = 30000

class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        higher_is_better: bool,
        data_dir: str, 
        output_dir: str,
    ):
        super().__init__()
        self.total_cost = 0.0
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.higher_is_better = higher_is_better
        self.data_preview: str | None = None
        self.data_dir = data_dir
        self.output_dir = output_dir

        # history track
        self.cost_history = []  # Track cost history for each action
        self.conversation_history = []

        # LLM settings
        self.llm_config = self.acfg.code
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')
        self.total_tokens = 0

        # Initialize model client
        self.model_client = ChatClient(
            model_name=self.llm_config.model_name,
            model_category=self.llm_config.model_mode,
            api_idx=self.llm_config.api_idx,
            port=self.llm_config.port,
            api_key=self.llm_config.api_key
        )
        self.model_settings = ModelSettings(
            max_completion_tokens=self.llm_config.max_completion_tokens,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p
        )

        # Pattern analysis components
        self.solution_analyzer = SolutionAnalyzer()
        self.pattern_discovery = PatternDiscovery()
        self.insight_generator = InsightGenerator()
        self._theme_cache = None
        self._theme_cache_size = 0
        
        # Token management - reserve tokens for task desc, instructions, etc.
        # Claude has 200K input limit, must be VERY conservative
        self.max_memory_tokens = 15000  # Strict limit for Memory section
        self.max_learned_patterns_tokens = 3000  # Limit for learned patterns
        self.max_data_preview_tokens = 8000  # Limit for data preview
        self.max_total_prompt_tokens = 150000  # Leave large buffer for safety

    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}, using rough estimate")
            # Rough estimate: ~4 chars per token
            return len(text) // 4
    
    def _generate_truncated_summary(self, max_tokens: int = None) -> str:
        """Generate a summary that fits within token budget.
        
        Prioritizes recent and high-scoring solutions.
        """
        if max_tokens is None:
            max_tokens = self.max_memory_tokens
        
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            return ""
        
        # Sort by score (descending) and recency (descending)
        # Give higher weight to both score and recency
        sorted_nodes = sorted(
            good_nodes,
            key=lambda n: (n.metric.value if hasattr(n, 'metric') else 0.0, n.step if hasattr(n, 'step') else 0),
            reverse=True
        )
        
        summary_parts = []
        current_tokens = 0
        nodes_included = 0
        
        for node in sorted_nodes:
            # Generate summary for this node (without code to save tokens)
            # Truncate plan if it's too long
            plan_text = node.plan[:500] + "..." if len(node.plan) > 500 else node.plan
            node_summary = f"Design: {plan_text}\n"
            
            if node.feedback is not None:
                # Truncate feedback aggressively - it can be very long
                feedback_str = str(node.feedback)
                if len(feedback_str) > 200:
                    feedback_str = feedback_str[:200] + "...[truncated]"
                node_summary += f"Execution Feedback: {feedback_str}\n"
            
            if node.raw_score is not None:
                node_summary += f"Test Metric Score: {node.raw_score}\n"
            if node.position_score is not None:
                node_summary += f"Test Position Score: {node.position_score}\n"
            
            node_tokens = self._count_tokens(node_summary)
            
            # Check if adding this node would exceed limit
            if current_tokens + node_tokens > max_tokens:
                # If we haven't included any nodes yet, include at least one (truncated)
                if nodes_included == 0:
                    # Drastically truncate to fit
                    node_summary = f"Design: {node.plan[:200]}...\n"
                    if node.position_score is not None:
                        node_summary += f"Position Score: {node.position_score}\n"
                    summary_parts.append(node_summary)
                    nodes_included += 1
                break
            
            summary_parts.append(node_summary)
            current_tokens += node_tokens
            nodes_included += 1
        
        # Add summary statistics
        total_nodes = len(good_nodes)
        if nodes_included < total_nodes:
            summary_parts.append(
                f"\n[Showing {nodes_included} of {total_nodes} solutions. "
                f"Omitted {total_nodes - nodes_included} lower-scoring solutions to manage context length.]\n"
            )
        
        return "\n-------------------------------\n".join(summary_parts)
    
    def _estimate_prompt_tokens(self, prompt_dict: Dict[str, Any]) -> int:
        """Estimate total tokens in a prompt dictionary."""
        prompt_str = compile_prompt_to_md(prompt_dict)
        return self._count_tokens(prompt_str)
    
    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node
    
    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks, use PyTorch rather than TensorFlow. You have access to a GPU and can use CUDA for faster computations if needed."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            # "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the script before finishing the code.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            f'All the provided input data is stored in {self.data_dir} directory.',
            f'**Save test predictions to `submission.csv` in {self.output_dir} directory as specified in the task description.** This file is critical for evaluation and scoring.',
            'Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.',
            'Your goal is to achieve the highest score, so ensure your code produces this file correctly.',
            f'You can also use the "{self.output_dir}/working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
    
    # TODO: AIDE currently **only support** gpt models, like gpt-4o-mini, can enhance later 
    def query_llm(self, system_message: str, user_message: str = None) -> str:
        """Query the LLM model."""
        system_message=compile_prompt_to_md(system_message) if system_message else None
        user_message=compile_prompt_to_md(user_message) if user_message else None
        
        # If no user message provided but we have a system message, treat the system message as user message
        # This is common in AIDE where the entire prompt is passed as system_message
        if system_message and not user_message:
            user_message = system_message
            system_message = None
        
        messages = opt_messages_to_list(system_message, user_message)
        # Check if messages conform to OpenAI standard message format
        # Each message should be a dict with 'role' and 'content' keys
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
            
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role: {msg['role']}. Role must be one of: system, user, assistant")
            
            if not isinstance(msg['content'], str):
                raise ValueError("Message content must be a string")
        
        # Token validation to prevent API errors
        total_tokens = sum(self._count_tokens(msg['content']) for msg in messages)
        if total_tokens > self.max_total_prompt_tokens:
            logger.error(f"Prompt exceeds token limit: {total_tokens} > {self.max_total_prompt_tokens}")
            raise ValueError(
                f"Prompt is too long: {total_tokens} tokens exceeds limit of {self.max_total_prompt_tokens}. "
                "This usually means the Memory section is too large. Consider reducing max_memory_tokens."
            )
        
        logger.info(f"Sending prompt with {total_tokens} tokens to LLM")
        return self.model_client.chat_completion(messages, self.model_settings)
    
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        # Log prompt structure for debugging
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        logger.info(f"plan_and_code_query: Estimated {prompt_tokens} tokens in prompt")
        
        # Log breakdown of sections
        if isinstance(prompt, dict):
            for key, value in prompt.items():
                if isinstance(value, (str, dict)):
                    section_str = compile_prompt_to_md({key: value})
                    section_tokens = self._count_tokens(section_str)
                    logger.info(f"  Section '{key}': {section_tokens} tokens")
                    if section_tokens > 30000:
                        logger.warning(f"  ⚠️ Section '{key}' is very large: {section_tokens} tokens!")
        
        completion_text = None
        for _ in range(retries):
            completion_text, cost = self.query_llm(
                system_message=prompt,
                user_message=None,
            )
            self.total_cost += cost
            self.cost_history.append({"action": "plan_and_code_query", "cost": cost})
            self.conversation_history.append({"role": "assistant", "content": completion_text})

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code, completion_text

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "","", completion_text  # type: ignore

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self._generate_truncated_summary(),
            "Instructions": {},
        }
        
        # Add learned patterns if we have enough data
        learned_patterns = self._get_learned_patterns_section()
        if learned_patterns:
            prompt.update(learned_patterns)
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric according to the task description.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                f"Your Python code should be able to generate a `submission.csv` file in the `{self.output_dir}` directory to get the metric score."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        self.conversation_history.append({
            "Type": "Draft",
            "Memory": prompt["Memory"],
            "Instructions": prompt["Instructions"],
        })

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        
        # Final token check before sending
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        if estimated_tokens > self.max_total_prompt_tokens:
            logger.warning(f"Draft prompt too large ({estimated_tokens} tokens), further reducing Memory...")
            # Emergency reduction: cut Memory in half
            prompt["Memory"] = self._generate_truncated_summary(max_tokens=self.max_memory_tokens // 2)
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, instruction_prompt=compile_prompt_to_md(prompt), node_type="draft", assistant=assistant)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance, i.e. the position score provided by the competition. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self._generate_truncated_summary(),
            "Instructions": {},
        }
        
        # Add learned patterns if we have enough data
        learned_patterns = self._get_learned_patterns_section()
        if learned_patterns:
            prompt.update(learned_patterns)
        
        # Truncate code if it's exceptionally long (usually shouldn't be, but just in case)
        code_to_show = parent_node.code
        if len(parent_node.code) > 20000:  # ~5000 tokens
            code_to_show = parent_node.code[:20000] + "\n\n# [Code truncated due to length...]"
            logger.warning(f"Parent node code truncated from {len(parent_node.code)} to 20000 chars")
        
        prompt["Previous solution"] = {
            "Code": wrap_code(code_to_show),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        self.conversation_history.append({
            "Type": "Improve",
            "Memory": prompt["Memory"],
            "Instructions": prompt["Instructions"],
        })

        # Final token check before sending
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        if estimated_tokens > self.max_total_prompt_tokens:
            logger.warning(f"Improve prompt too large ({estimated_tokens} tokens), reducing Memory and removing Learned Patterns...")
            # Emergency reduction
            prompt["Memory"] = self._generate_truncated_summary(max_tokens=self.max_memory_tokens // 2)
            # Remove learned patterns if still too large
            if "Learned Patterns" in prompt:
                del prompt["Learned Patterns"]

        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            instruction_prompt=compile_prompt_to_md(prompt),
            node_type="improve",
            assistant=assistant
        )

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.feedback, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        self.conversation_history.append({
            "Type": "Debug",
            "Instructions": prompt["Instructions"],
        })

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node, instruction_prompt=compile_prompt_to_md(prompt), node_type="debug", assistant=assistant)

    def _extract_themes_from_solutions(self, nodes: List[Node], is_best: bool = True) -> Dict[str, Any]:
        """Extract common themes from a set of solutions."""
        if not nodes:
            return {}
        
        themes = {
            'model_types': [],
            'preprocessing': [],
            'validation_strategies': [],
            'hyperparameter_methods': [],
            'plan_keywords': [],
            'avg_score': 0.0,
            'code_complexity': 0,
            'has_ensemble': 0,
        }
        
        # Analyze each solution
        for node in nodes:
            try:
                features = self.solution_analyzer.analyze_solution(node)
                themes['model_types'].extend(features['model_type'])
                themes['preprocessing'].extend(features['preprocessing'])
                themes['validation_strategies'].append(features['validation_strategy'])
                themes['hyperparameter_methods'].append(features['hyperparameter_method'])
                themes['plan_keywords'].extend(features['plan_keywords'])
                themes['code_complexity'] += features['code_length']
                themes['has_ensemble'] += 1 if features['has_ensemble'] else 0
            except Exception as e:
                logger.warning(f"Failed to extract themes from node {node.id}: {e}")
                continue
        
        # Compute statistics
        if nodes:
            from collections import Counter
            themes['model_types'] = [item for item, count in Counter(themes['model_types']).most_common(3)]
            themes['preprocessing'] = [item for item, count in Counter(themes['preprocessing']).most_common(5)]
            themes['validation_strategies'] = Counter(themes['validation_strategies']).most_common(1)[0][0] if themes['validation_strategies'] else 'unknown'
            themes['hyperparameter_methods'] = Counter(themes['hyperparameter_methods']).most_common(1)[0][0] if themes['hyperparameter_methods'] else 'fixed_params'
            themes['plan_keywords'] = [item for item, count in Counter(themes['plan_keywords']).most_common(5)]
            themes['avg_score'] = sum(n.metric.value for n in nodes if hasattr(n, 'metric')) / len(nodes)
            themes['code_complexity'] = themes['code_complexity'] / len(nodes)
            themes['has_ensemble'] = themes['has_ensemble'] / len(nodes)
        
        return themes
    
    def _analyze_best_vs_worst(self) -> str:
        """Compare best 3 vs worst 3 solutions and generate guidance."""
        good_nodes = self.journal.good_nodes
        
        if len(good_nodes) < 6:
            return ""
        
        # Sort by score
        sorted_nodes = sorted(good_nodes, key=lambda n: n.metric.value if hasattr(n, 'metric') else 0.0, reverse=True)
        
        best_3 = sorted_nodes[:3]
        worst_3 = sorted_nodes[-3:]
        
        # Extract themes
        best_themes = self._extract_themes_from_solutions(best_3, is_best=True)
        worst_themes = self._extract_themes_from_solutions(worst_3, is_best=False)
        
        # Generate comparative analysis
        analysis_parts = []
        analysis_parts.append("**Learned Patterns from Recent Solutions:**\n")
        
        # High-performing approaches
        if best_themes:
            analysis_parts.append(f"**High-Performing Approaches** (Top 3 solutions averaging {best_themes['avg_score']:.3f} score):")
            
            if best_themes['model_types']:
                models_str = ', '.join(best_themes['model_types'])
                analysis_parts.append(f"- Primary models: **{models_str}**")
            
            if best_themes['preprocessing']:
                prep_str = ', '.join(best_themes['preprocessing'])
                analysis_parts.append(f"- Preprocessing techniques: {prep_str}")
            
            if best_themes['validation_strategies'] != 'unknown':
                analysis_parts.append(f"- Validation strategy: {best_themes['validation_strategies']}")
            
            if best_themes['plan_keywords']:
                keywords_str = ', '.join(f'"{kw}"' for kw in best_themes['plan_keywords'][:3])
                analysis_parts.append(f"- Key concepts in plans: {keywords_str}")
            
            if best_themes['has_ensemble'] > 0.5:
                analysis_parts.append(f"- Uses ensemble methods")
        
        analysis_parts.append("")
        
        # Low-performing approaches
        if worst_themes:
            analysis_parts.append(f"**Low-Performing Approaches** (Bottom 3 solutions averaging {worst_themes['avg_score']:.3f} score):")
            
            if worst_themes['model_types']:
                models_str = ', '.join(worst_themes['model_types'])
                analysis_parts.append(f"- Primary models: {models_str}")
            
            if worst_themes['preprocessing']:
                prep_str = ', '.join(worst_themes['preprocessing'])
                analysis_parts.append(f"- Preprocessing: {prep_str}")
            else:
                analysis_parts.append(f"- Minimal or no preprocessing")
            
            if worst_themes['validation_strategies'] != 'unknown':
                analysis_parts.append(f"- Validation: {worst_themes['validation_strategies']}")
        
        analysis_parts.append("")
        
        # Key differences
        analysis_parts.append("**Key Differences:**")
        
        # Model comparison
        best_models = set(best_themes.get('model_types', []))
        worst_models = set(worst_themes.get('model_types', []))
        unique_best = best_models - worst_models
        unique_worst = worst_models - best_models
        
        if unique_best:
            analysis_parts.append(f"- Top solutions use {', '.join(unique_best)} which bottom solutions don't")
        if unique_worst:
            analysis_parts.append(f"- Bottom solutions use {', '.join(unique_worst)} which top solutions avoid")
        
        # Preprocessing comparison
        best_prep = set(best_themes.get('preprocessing', []))
        worst_prep = set(worst_themes.get('preprocessing', []))
        if len(best_prep) > len(worst_prep):
            analysis_parts.append(f"- Top solutions use more preprocessing steps ({len(best_prep)} vs {len(worst_prep)})")
        
        # Code complexity comparison
        if best_themes.get('code_complexity', 0) > worst_themes.get('code_complexity', 0) * 1.3:
            analysis_parts.append(f"- Top solutions are more comprehensive ({best_themes['code_complexity']:.0f} vs {worst_themes['code_complexity']:.0f} lines)")
        
        analysis_parts.append("")
        
        # Recommendations
        analysis_parts.append("**Recommendations:**")
        
        if best_themes.get('model_types'):
            analysis_parts.append(f"- Consider using {best_themes['model_types'][0]} as the primary approach")
        
        if best_themes.get('preprocessing'):
            analysis_parts.append(f"- Include preprocessing: {', '.join(best_themes['preprocessing'][:3])}")
        
        if best_themes.get('validation_strategies') and best_themes['validation_strategies'] != worst_themes.get('validation_strategies'):
            analysis_parts.append(f"- Use {best_themes['validation_strategies']} for validation")
        
        if worst_models and not best_models.intersection(worst_models):
            analysis_parts.append(f"- Avoid relying solely on {', '.join(list(worst_models)[:2])}")
        
        return "\n".join(analysis_parts)
    
    def _get_learned_patterns_section(self) -> Dict[str, str]:
        """Generate 'Learned Patterns' section for prompts."""
        # Check if we need to update cache
        current_size = len(self.journal.good_nodes)
        
        if self._theme_cache is None or current_size != self._theme_cache_size:
            # Update cache
            if current_size >= 6:
                full_analysis = self._analyze_best_vs_worst()
                # Truncate to fit token budget
                self._theme_cache = self._truncate_text_to_tokens(full_analysis, self.max_learned_patterns_tokens)
                self._theme_cache_size = current_size
            else:
                self._theme_cache = ""
                self._theme_cache_size = current_size
        
        if self._theme_cache:
            return {"Learned Patterns": self._theme_cache}
        else:
            return {}
    
    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return text
        
        current_tokens = self._count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Binary search to find the right length
        # Rough estimate: keep proportion of characters
        target_length = int(len(text) * (max_tokens / current_tokens) * 0.9)  # 0.9 safety factor
        truncated = text[:target_length]
        truncated += f"\n\n[... Truncated {current_tokens - max_tokens} excess tokens to fit context limit ...]"
        
        return truncated
    
    def update_data_preview(
        self,
    ):
        full_preview = data_preview.generate(self.cfg.workspace_dir)
        # Truncate data preview to fit token budget
        self.data_preview = self._truncate_text_to_tokens(full_preview, self.max_data_preview_tokens)
    
    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        # TODO: need to check if LLM call is working appropriately later
        if parent_node is None:
            print("Drafting...")
            result_node = self._draft()
        elif parent_node.is_buggy:
            print("Debugging...")
            result_node = self._debug(parent_node)
        else:
            print("Improving...")
            result_node = self._improve(parent_node)

        self.parse_exec_result(node=result_node, exec_result=exec_callback(result_node.code))
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: Dict):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        eval_result = ExecutionResult(
            status=exec_result["action_status"],
            feedback=exec_result["feedback"],
            raw_score=exec_result["current_raw_score"],
            position_score=exec_result["current_position_score"],
            # best_raw_score=exec_result["best_raw_score"],
            # best_position_score=exec_result["best_position_score"],
        )
        node.absorb_exec_result(eval_result)
        node.analysis = ""
        node.is_buggy = eval_result.status == "FAILED"

        if node.is_buggy:
            node.metric = WorstMetricValue(value = 0.0)
        else:
            node.metric = MetricValue(eval_result.position_score, maximize=True)