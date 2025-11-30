"""
MLE-STAR Agent: Machine Learning Engineering via Search and Targeted Refinement
Extends AIDE with MLE-STAR methodology including web search, HPO, refinement, and ablations.
"""
import logging
import random
import humanize
import json
import re
import os
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import tiktoken
from mledojo.agent.aide.agent import Agent as AIDEAgent
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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI package not available. Perplexity web search will be disabled.")

logger = logging.getLogger("mlestar")

ExecCallbackType = Callable[[str], Dict]


@dataclass
class MLEStarConfig:
    """Configuration for MLE-STAR specific settings"""
    search_iterations: int = 3
    refinement_iterations: int = 5
    perplexity_model: str = "llama-3.1-sonar-large-128k-online"
    enable_web_search: bool = True
    enable_ablation: bool = True
    enable_refinement: bool = True
    enable_ensemble: bool = True


class MLESTARAgent(AIDEAgent):
    """
    MLE-STAR Agent that extends AIDE with:
    - Web search (Perplexity) for model retrieval
    - HPO capabilities
    - Targeted refinement with ablations
    - Ensemble strategies
    """
    
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        higher_is_better: bool,
        data_dir: str,
        output_dir: str,
        mlestar_cfg: Optional[MLEStarConfig] = None,
    ):
        """
        Initialize MLE-STAR Agent.
        
        Args:
            task_desc: Task description
            cfg: Configuration object
            journal: Journal for tracking experiments
            higher_is_better: Whether higher scores are better
            data_dir: Data directory path
            output_dir: Output directory path
            mlestar_cfg: MLE-STAR specific configuration
        """
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.higher_is_better = higher_is_better
        self.data_preview: Optional[str] = None
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mlestar_cfg = mlestar_cfg or MLEStarConfig()
        
        # State tracking
        self.total_cost = 0.0
        self.cost_history = []
        self.conversation_history = []
        
        # MLE-STAR specific state
        self.retrieved_models: List[Dict] = []
        self.initial_solutions: List[Node] = []  # List of Node objects
        self.best_solution: Optional[str] = None
        self.best_score: float = -float('inf') if higher_is_better else float('inf')
        self.ablation_history: List[Dict] = []
        self.refined_code_blocks: List[Dict] = []
        self.refinement_plans: List[Tuple[str, float]] = []  # (plan, score)
        self.ensemble_solutions: List[str] = []
        self.ensemble_plans: List[Tuple[str, float]] = []
        
        # Tracking for graphs and errors
        self.reward_history: List[Dict] = []  # [{"timestep": int, "reward": float, "phase": str, "iteration": int, "error": Optional[str]}]
        self.error_history: List[Dict] = []  # [{"timestep": int, "phase": str, "error": str, "traceback": str}]
        self.timestep = 0
        
        # Bug history to avoid loops (like AIDE journal)
        self.bug_history: List[Dict] = []  # [{"code_hash": str, "error": str, "attempt": int, "fix_attempted": str, "code_snippet": str}]
        
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
        
        # Initialize web search client
        self._init_web_client()
        
        # Phase tracking
        self.current_phase = "search"  # search, foundation, refinement, ensemble, validation
    
    def _init_web_client(self):
        """Initialize web search client (Perplexity)."""
        self.web_client = None
        # Support both PPLX_API_KEY (Alxandria) and PERPLEXITY_API_KEY (MLE-STAR)
        perplexity_key = os.getenv('PPLX_API_KEY') or os.getenv('PERPLEXITY_API_KEY')
        if perplexity_key and OpenAI:
            try:
                self.web_client = OpenAI(
                    api_key=perplexity_key,
                    base_url="https://api.perplexity.ai"
                )
                logger.info("Initialized Perplexity web client")
            except Exception as e:
                logger.error(f"Failed to initialize Perplexity client: {str(e)}")
        else:
            if not OpenAI:
                logger.warning("OpenAI package not available. Web search disabled.")
            else:
                logger.warning("Perplexity API key not found. Web search disabled.")
    
    def query_llm(self, system_message: str, user_message: Optional[str] = None) -> Tuple[str, float]:
        """Query the LLM model."""
        system_message = compile_prompt_to_md(system_message) if system_message else None
        user_message = compile_prompt_to_md(user_message) if user_message else None
        messages = opt_messages_to_list(system_message, user_message)
        
        # chat_completion returns (response_text, cost) tuple
        result = self.model_client.chat_completion(messages, self.model_settings)
        
        # Safely unpack the tuple
        if isinstance(result, tuple) and len(result) >= 2:
            response, cost = result[0], result[1]
        elif isinstance(result, tuple) and len(result) == 1:
            response, cost = result[0], 0.0
        else:
            logger.error(f"chat_completion returned unexpected type: {type(result)}, value: {result}")
            response = str(result) if result else ""
            cost = 0.0
        
        self.total_cost += cost
        
        # Ensure response is a string (handle nested tuples and edge cases)
        if isinstance(response, tuple):
            logger.debug(f"query_llm: response is nested tuple, extracting first element")
            response = response[0] if len(response) > 0 else ""
        
        if not isinstance(response, str):
            # Only log as warning if it's unexpected (not None)
            if response is not None:
                logger.debug(f"query_llm: converting non-string response {type(response)} to string")
            response = str(response) if response else ""
        
        return response, cost
    
    def _safe_query_llm(self, system_message: str, user_message: Optional[str] = None) -> str:
        """Safely query LLM and return just the response string."""
        result = self.query_llm(system_message, user_message)
        if isinstance(result, tuple):
            response, _ = result
        else:
            logger.warning(f"_safe_query_llm: query_llm returned non-tuple: {type(result)}")
            response = str(result) if result else ""
        
        # Ensure response is a string
        if not isinstance(response, str):
            logger.warning(f"_safe_query_llm: response is not a string: {type(response)}")
            response = str(response) if response else ""
        
        return response
    
    def search_web(self, query: str) -> str:
        """Search web using Perplexity API (synchronous for Jupyter compatibility)."""
        if not self.web_client:
            logger.warning("Web search not available, returning empty result")
            return ""
        
        try:
            # Use synchronous call instead of asyncio for Jupyter compatibility
            perplexity_model = self.mlestar_cfg.perplexity_model
            response = self.web_client.chat.completions.create(
                model=perplexity_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that performs web searches."},
                    {"role": "user", "content": query}
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            if response and response.choices:
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
        return ""
    
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        response_text = ""
        for attempt in range(retries):
            try:
                result = self.query_llm(
                    system_message=prompt,
                    user_message=None,
                )
                # query_llm returns (response, cost) tuple
                if isinstance(result, tuple) and len(result) >= 2:
                    response, cost = result[0], result[1]
                elif isinstance(result, tuple) and len(result) == 1:
                    response, cost = result[0], 0.0
                else:
                    response = str(result) if result else ""
                    cost = 0.0
                
                self.total_cost += cost
                self.cost_history.append({"action": "plan_and_code_query", "cost": cost})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Ensure response is a string
                if not isinstance(response, str):
                    response = str(response) if response else ""
                
                response_text = response
                code = extract_code(response)
                nl_text = extract_text_up_to_code(response)

                if code and nl_text:
                    # merge all code blocks into a single string
                    return nl_text, code, response
                elif code:
                    # If we have code but no natural language, use empty string for plan
                    return "", code, response
                elif nl_text:
                    # If we have text but no code, return it as plan
                    return nl_text, "", response

                if attempt < retries - 1:
                    logger.debug(f"Plan + code extraction failed, retrying... (attempt {attempt + 1}/{retries})")
            except Exception as e:
                logger.error(f"Error in plan_and_code_query attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    break
        
        logger.warning("Final plan + code extraction attempt failed, giving up...")
        return "", "", response_text
    
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
    
    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
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
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
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

    def update_data_preview(self):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)
    
    def parse_exec_result(self, node: Node, exec_result: Dict):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        eval_result = ExecutionResult(
            status=exec_result["action_status"],
            feedback=exec_result["feedback"],
            raw_score=exec_result["current_raw_score"],
            position_score=exec_result["current_position_score"],
        )
        node.absorb_exec_result(eval_result)
        node.analysis = ""
        node.is_buggy = eval_result.status == "FAILED"

        if node.is_buggy:
            node.metric = WorstMetricValue(value = 0.0)
        else:
            node.metric = MetricValue(eval_result.position_score, maximize=True)
        
    def _web_search_phase(self) -> List[Dict[str, str]]:
        """
        Phase 1: Web search for models (MLE-STAR Prompt 1).
        Uses Perplexity to search for effective models.
        """
        if not self.web_client or not self.mlestar_cfg.enable_web_search:
            logger.warning("Web search not available, skipping web search phase. Will proceed with standard AIDE workflow.")
            return []
        
        logger.info(f"Starting web search phase ({self.mlestar_cfg.search_iterations} iterations)")
        all_models = []
        failed_searches = 0
        
        for i in range(self.mlestar_cfg.search_iterations):
            logger.info(f"Web search iteration {i+1}/{self.mlestar_cfg.search_iterations}")
            try:
                # Use MLE-STAR Prompt 1
                prompt = f"""# Competition
{self.task_desc}

# Your task
- List 5 recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""
                
                search_result = self.search_web(prompt)
                if search_result and not search_result.startswith('Error:'):
                    models = self._parse_models_from_response(search_result, num_models=5)
                    if models and len(models) > 0:
                        all_models.extend(models)
                    else:
                        failed_searches += 1
                        logger.warning(f"Search iteration {i+1} returned no valid models")
                else:
                    failed_searches += 1
                    logger.warning(f"Search iteration {i+1} failed")
            except Exception as e:
                failed_searches += 1
                logger.error(f"Search iteration {i+1} failed: {e}")
        
        if failed_searches == self.mlestar_cfg.search_iterations:
            logger.warning("All web searches failed. Proceeding with standard AIDE workflow.")
            return []
        
        # Deduplicate models by name
        seen_names = set()
        unique_models = []
        for model in all_models:
            if model.get('model_name') and model['model_name'] not in seen_names:
                seen_names.add(model['model_name'])
                unique_models.append(model)
        
        self.retrieved_models = unique_models[:10] if unique_models else []  # Keep top 10
        logger.info(f"Retrieved {len(self.retrieved_models)} unique models")
        return self.retrieved_models
    
    def _parse_models_from_response(self, content: str, num_models: int) -> List[Dict[str, str]]:
        """Parse model information from search response."""
        models = []
        if not content:
            logger.warning("Empty response from web search")
            return self._create_placeholder_models(num_models)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate structure
                    valid_models = []
                    for item in parsed:
                        if isinstance(item, dict) and 'model_name' in item and 'example_code' in item:
                            valid_models.append({
                                'model_name': str(item['model_name']),
                                'example_code': str(item['example_code'])
                            })
                    if valid_models:
                        return valid_models[:num_models]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"JSON parsing failed: {e}")
        
        # Fallback: try to extract model names and code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
        model_names = re.findall(r'model[_\s]*name[:\s]*["\']?([^"\'\n]+)["\']?', content, re.IGNORECASE)
        
        # Also try to find model names in markdown lists or numbered lists
        if not model_names:
            model_names = re.findall(r'(?:^|\n)\s*(?:[-*]|\d+\.)\s*([A-Za-z][^:\n]+?)(?::|model|approach)', content, re.MULTILINE | re.IGNORECASE)
        
        for i, (name, code) in enumerate(zip(model_names[:num_models], code_blocks[:num_models])):
            models.append({
                'model_name': name.strip() if name else f"Model_{i+1}",
                'example_code': code.strip() if code else f"# Code for {name or f'Model_{i+1}'}"
            })
        
        # If we have code blocks but no names, use generic names
        if code_blocks and not model_names:
            for i, code in enumerate(code_blocks[:num_models]):
                models.append({
                    'model_name': f"Model_{i+1}",
                    'example_code': code.strip()
                })
        
        # If we still don't have models, create placeholder entries
        return models[:num_models] if models else self._create_placeholder_models(num_models)
    
    def _create_placeholder_models(self, num_models: int) -> List[Dict[str, str]]:
        """Create placeholder model entries when parsing fails."""
        return [{
            'model_name': f"Model_{i+1}",
            'example_code': f"# Model {i+1} code not found in search results\n# Please implement based on task description"
        } for i in range(num_models)]
    
    def _generate_initial_solution(self, model_info: Dict[str, str]) -> Optional[Node]:
        """
        Generate initial solution using a retrieved model (MLE-STAR Prompt 2).
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "We will now provide a task description and a model description.\n"
                "You need to implement your Python solution using the provided model."
            ),
            "Task description": self.task_desc,
            "Model description": {
                "Model name": model_info['model_name'],
                "Example Python code": wrap_code(model_info['example_code'])
            },
            "Instructions": {
                "Your task": [
                    "Implement the solution in Python.",
                    "You must use the model as described in the model description.",
                    "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                    "Propose an evaluation metric that is reasonable for this task.",
                    "All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                    "Do not include other models that are not directly related to the model described.",
                    "Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed.",
                    "The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.",
                    "Only use the provided train data in the `./input` directory. Do not load test data.",
                    "If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run."
                ],
                "Required": [
                    "There should be no additional headings or text in your response.",
                    "Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.",
                    "The code should be a single-file Python program that is self-contained and can be executed as-is.",
                    "Your response should only contain a single code block.",
                    "Do not use exit() function in the Python code.",
                    "Do not use try: and except: or if else to ignore unintended behavior."
                ]
            }
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment
        
        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        if code:
            return Node(
                plan=plan,
                code=code,
                instruction_prompt=compile_prompt_to_md(prompt),
                node_type="initial_solution",
                assistant=assistant
            )
        return None
    
    def _merge_solutions(self, base_solution: Node, reference_solution: Node) -> Optional[Node]:
        """
        Merge two solutions (MLE-STAR Prompt 3).
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "We will now provide a base solution and an additional reference solution.\n"
                "You need to implement your Python solution by integrating reference solution to the base solution."
            ),
            "Base solution": wrap_code(base_solution.code),
            "Reference solution": wrap_code(reference_solution.code),
            "Instructions": {
                "Your task": [
                    "Implement the solution in Python.",
                    "You have to integrate the reference solution to the base solution.",
                    "Your code base should be the base solution.",
                    "Try to train additional model of the reference solution.",
                    "When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).",
                    "When integrating, ensemble the models.",
                    "The solution design should be relatively simple.",
                    "The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.",
                    "Only use the provided train data in the `./input` directory. Do not load test data.",
                    "If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run."
                ],
                "Required": [
                    "There should be no additional headings or text in your response.",
                    "Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.",
                    "The code should be a single-file Python program that is self-contained and can be executed as-is.",
                    "Your response should only contain a single code block.",
                    "Do not use exit() function in the Python code.",
                    "Do not use try: and except: or if else to ignore unintended behavior."
                ]
            }
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        if code:
            return Node(
                plan=plan,
                code=code,
                parent=base_solution,
                instruction_prompt=compile_prompt_to_md(prompt),
                node_type="merged_solution",
                assistant=assistant
            )
        return None
    
    def _ablation_study(self, solution_node: Node, previous_ablations: List[Dict]) -> Optional[Node]:
        """
        Perform ablation study (MLE-STAR Prompts 4 & 5).
        """
        previous_ablation_text = ""
        for i, abl in enumerate(previous_ablations):
            previous_ablation_text += f"\n## Previous ablation study result {i}\n{abl.get('summary', '')}\n"
        
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "In order to win this competition, you need to perform an ablation study on the current Python solution "
                "to know which parts of the code contribute the most to the overall performance.\n"
                "We will now provide a current Python solution.\n"
                "We will also provide the summaries of previous ablation studies."
            ),
            "Python solution": wrap_code(solution_node.code),
            "Previous ablation studies": previous_ablation_text if previous_ablations else "None",
            "Instructions": {
                "Your task": [
                    "You need you to generate a simple Python code that performs an ablation study on the train.py script.",
                    "The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.",
                    "Your ablation study should concentrate on the other parts that have not been previously considered.",
                    "For each ablation, print out how the modification affects the model's performance."
                ],
                "Response format": [
                    "There should be no additional headings or text in your response.",
                    "The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.",
                    "The code should include a printing statement that shows the performance of each ablation.",
                    "The code should consequently print out what part of the code contributes the most to the overall performance."
                ]
            }
        }
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        if code:
            return Node(
                plan=plan,
                code=code,
                parent=solution_node,
                instruction_prompt=compile_prompt_to_md(prompt),
                node_type="ablation_study",
                assistant=assistant
            )
        return None
    
    def _summarize_ablation(self, ablation_code: str, ablation_results: str) -> Dict:
        """
        Summarize ablation study results (MLE-STAR Prompt 5).
        """
        prompt: Any = {
            "Your code for ablation study was": wrap_code(ablation_code),
            "Ablation study results after running the above code": wrap_code(ablation_results, lang=""),
            "Your task": "Summarize the result of ablation study based on the code and printed output."
        }
        
        summary, _, _ = self.plan_and_code_query(prompt)
        return {
            'summary': summary,
            'code': ablation_code,
            'results': ablation_results
        }
    
    def _extract_refinement_plan(self, solution_node: Node, ablation_summary: Dict, previous_refinements: List[Dict]) -> Optional[Dict]:
        """
        Extract code block and create refinement plan (MLE-STAR Prompt 6).
        """
        previous_refinement_text = ""
        for i, ref in enumerate(previous_refinements):
            previous_refinement_text += f"\n## Code block {i}\n{ref.get('code_block', '')}\n"
        
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "In order to win this competition, you need to extract a code block from the current Python solution "
                "and improve the extracted block for better performance.\n"
                "Your suggestion should be based on the ablation study results of the current Python solution.\n"
                "We will now provide the current Python solution and the ablation study results.\n"
                "We also provide code blocks which you have tried to improve previously."
            ),
            "Python solution": wrap_code(solution_node.code),
            "Ablation study results": ablation_summary.get('summary', ''),
            "Previous code blocks": previous_refinement_text if previous_refinements else "None",
            "Instructions": {
                "Your task": [
                    "Given the ablation study results, suggest an effective next plan to improve the above Python script.",
                    "The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).",
                    "Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).",
                    "Try to improve the other part which was not considered before.",
                    "Also extract the code block from the above Python script that need to be improved according to the proposed plan. You should try to extract the code block which was not improved before."
                ],
                "Response format": (
                    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) "
                    "and a single markdown code block which is the code block that need to be improved.\n"
                    "The code block can be long but should be exactly extracted from the Python script provided above.\n\n"
                    "Use this JSON schema:\n"
                    "Refine_Plan = {'code_block': str, 'plan': str}\n"
                    "Return: list[Refine_Plan]"
                )
            }
        }
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\[.*\]', code or plan, re.DOTALL)
            if json_match:
                refine_plans = json.loads(json_match.group())
                if isinstance(refine_plans, list) and len(refine_plans) > 0:
                    return refine_plans[0]
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: extract code block and plan from text
        code_block = extract_code(code or plan)
        plan_text = extract_text_up_to_code(code or plan) or plan
        
        if code_block:
            return {
                'code_block': code_block,
                'plan': plan_text
            }
        return None
    
    def _refine_code_block(self, code_block: str, improvement_plan: str) -> Optional[str]:
        """
        Refine a code block based on improvement plan (MLE-STAR Prompt 7).
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "In order to win this competition, you need refine the code block for better performance based on the improvement plan.\n"
                "We will now provide the code block and the improvement plan."
            ),
            "Code block": wrap_code(code_block),
            "Improvement plan": improvement_plan,
            "Instructions": {
                "Your task": [
                    "Implement the improvement plan on the above code block. But do not remove subsampling if exists.",
                    "The code block should be improved according to the proposed plan.",
                    "Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables."
                ],
                "Response format": (
                    "Your response should be a single markdown code block (wrapped in ```) which is the improved code block.\n"
                    "There should be no additional headings or text in your response."
                )
            }
        }
        
        _, code, _ = self.plan_and_code_query(prompt)
        return extract_code(code) if code else None
    
    def _suggest_refinement_plan(self, code_block: str, previous_plans: List[Dict]) -> str:
        """
        Suggest a new refinement plan when previous ones didn't work (MLE-STAR Prompt 8).
        """
        previous_plans_text = ""
        for i, plan_info in enumerate(previous_plans):
            previous_plans_text += f"\n## Plan: {plan_info.get('plan', '')}\n## Score: {plan_info.get('score', 'N/A')}\n"
        
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                "In order to win this competition, you have to improve the code block for better performance.\n"
                "We will provide the code block you are improving and the improvement plans you have tried."
            ),
            "Code block": wrap_code(code_block),
            "Improvement plans you have tried": previous_plans_text if previous_plans else "None",
            "Instructions": {
                "Your task": [
                    "Suggest a better plan to improve the above code block.",
                    "The suggested plan must be novel and effective.",
                    "Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).",
                    "The suggested plan should be differ from the previous plans you have tried and should receive a higher score."
                ],
                "Response format": (
                    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).\n"
                    "There should be no additional headings or text in your response."
                )
            }
        }
        
        plan, _, _ = self.plan_and_code_query(prompt)
        return plan or ""
    
    def _suggest_ensemble_plan(self, solutions: List[Node], previous_plans: List[Dict]) -> str:
        """
        Suggest ensemble plan (MLE-STAR Prompt 9).
        """
        solutions_text = ""
        for i, sol in enumerate(solutions):
            solutions_text += f"\n# {i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} Python Solution\n{wrap_code(sol.code)}\n"
        
        previous_plans_text = ""
        for i, plan_info in enumerate(previous_plans):
            previous_plans_text += f"\n## Plan: {plan_info.get('plan', '')}\n## Score: {plan_info.get('score', 'N/A')}\n"
        
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                f"In order to win this competition, you have to ensemble {len(solutions)} Python Solutions for better performance.\n"
                "We will provide the Python Solutions and the ensemble plans you have tried."
            ),
            "Solutions": solutions_text,
            "Ensemble plans you have tried": previous_plans_text if previous_plans else "None",
            "Instructions": {
                "Your task": [
                    f"Suggest a better plan to ensemble the {len(solutions)} solutions. You should concentrate how to merge, not the other parts like hyperparameters.",
                    "The suggested plan must be easy to implement, novel, and effective.",
                    "The suggested plan should be differ from the previous plans you have tried and should receive a higher (or lower) score."
                ],
                "Response format": (
                    "Your response should be an outline/sketch of your proposed solution in natural language.\n"
                    "There should be no additional headings or text in your response.\n"
                    "Plan should not modify the original solutions too much since execution error can occur."
                )
            }
        }
        
        plan, _, _ = self.plan_and_code_query(prompt)
        return plan or ""
    
    def _ensemble_solutions(self, solutions: List[Node], ensemble_plan: str) -> Optional[Node]:
        """
        Create ensemble solution (MLE-STAR Prompt 10).
        """
        solutions_text = ""
        for i, sol in enumerate(solutions):
            solutions_text += f"\n# {i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} Python Solution\n{wrap_code(sol.code)}\n"
        
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition.\n"
                f"In order to win this competition, you need to ensemble {len(solutions)} Python Solutions for better performance based on the ensemble plan.\n"
                "We will now provide the Python Solutions and the ensemble plan."
            ),
            "Solutions": solutions_text,
            "Ensemble Plan": ensemble_plan,
            "Instructions": {
                "Your task": [
                    "Implement the ensemble plan with the provided solutions.",
                    "Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much.",
                    "All the provided data (except previous submissions; do not load submissions) is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                    "The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set."
                ],
                "Response format required": [
                    "Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {len(solutions)} Python Solutions.",
                    "There should be no additional headings or text in your response.",
                    "Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {len(solutions)} provided solutions.",
                    "Do not forget the `./final/submission.csv` file.",
                    "Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.",
                    "The code should be a single-file Python program that is self-contained and can be executed as-is."
                ]
            }
        }
        
        _, code, assistant = self.plan_and_code_query(prompt)
        if code:
            # Use the best solution as parent
            best_solution = max(solutions, key=lambda n: n.metric.value if n.metric else 0)
            return Node(
                plan=ensemble_plan,
                code=extract_code(code),
                parent=best_solution,
                instruction_prompt=compile_prompt_to_md(prompt),
                node_type="ensemble",
                assistant=assistant
            )
        return None
    
    def _check_data_leakage(self, code: str) -> Dict:
        """
        Check for data leakage (MLE-STAR Prompts 12 & 13).
        """
        prompt: Any = {
            "Python code": wrap_code(code),
            "Instructions": {
                "Your task": [
                    "Extract the code block where the validation and test samples are preprocessed using training samples.",
                    "Check that the model is trained with only training samples.",
                    "Check that before printing the final validation score, the model is not trained the validation samples.",
                    "Also check whether the validation and test samples are preprocessed correctly, preventing information from the validation or test samples from influencing the training process (i.e., preventing data leakage)."
                ],
                "Requirement": [
                    "Extract a code block and also check the data leakage.",
                    "The code block should be an exact subset of the above Python code.",
                    "Your response for a code block should be a single markdown code block.",
                    "If data leakage is present on validation and test samples, answer 'Yes Data Leakage'.",
                    "If data leakage is not present on validation and test samples, answer 'No Data Leakage'.",
                    "Use this JSON schema:\n"
                    "Answer = {'leakage_status': str, 'code_block': str}\n"
                    "Return: list[Answer]"
                ]
            }
        }
        
        _, response, _ = self.plan_and_code_query(prompt)
        
        # Try to parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                answers = json.loads(json_match.group())
                if isinstance(answers, list) and len(answers) > 0:
                    return answers[0]
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback parsing
        leakage_status = "No Data Leakage"
        if "Yes Data Leakage" in response or "data leakage" in response.lower():
            leakage_status = "Yes Data Leakage"
        
        code_block = extract_code(response)
        return {
            'leakage_status': leakage_status,
            'code_block': code_block
        }
    
    def _fix_data_leakage(self, code: str) -> Optional[str]:
        """
        Fix data leakage issues (MLE-STAR Prompt 13).
        """
        prompt: Any = {
            "Python code": wrap_code(code),
            "Instructions": {
                "Your task": [
                    "In the above Python code, the validation and test samples are influencing the training process, i.e., not correctly preprocessed.",
                    "Ensure that the model is trained with only training samples.",
                    "Ensure that before printing the final validation score, the model is not trained on the validation samples.",
                    "Refine the code to prevent such data leakage problem."
                ],
                "Requirement": [
                    "Your response should be a single markdown code block.",
                    "Note that all the variables are defined earlier. Just modify it with the above code."
                ]
            }
        }
        
        _, code, _ = self.plan_and_code_query(prompt)
        return extract_code(code) if code else None
    
    def _check_data_usage(self, code: str, task_desc: str) -> Optional[str]:
        """
        Check if all provided information is used (MLE-STAR Prompt 14).
        """
        prompt: Any = {
            "Solution Code": wrap_code(code),
            "Task description": task_desc,
            "Instructions": {
                "Your task": [
                    "If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.",
                    "DO NOT USE TRY and EXCEPT; just occur error so we can debug it!",
                    "See the task description carefully, to know how to extract unused information effectively.",
                    "When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {final_validation_score}' as in original solution code."
                ],
                "Response format": (
                    "Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. "
                    "There should be no additional headings or text in your response.\n"
                    "Option 2: If the code used all the provided information, simply state that 'All the provided information is used.'"
                )
            }
        }
        
        plan, code, _ = self.plan_and_code_query(prompt)
        
        if "All the provided information is used" in plan or "All the provided information is used" in (code or ""):
            return None  # No changes needed
        
        return extract_code(code) if code else None
    
    def step(self, exec_callback: ExecCallbackType):
        """
        MLE-STAR workflow step that orchestrates the different phases.
        """
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        
        # Phase 1: Web Search (only on first step)
        if self.current_phase == "search" and not self.retrieved_models:
            logger.info("Phase 1: Web Search")
            models = self._web_search_phase()
            if not models:
                # If web search failed or returned no models, skip to standard AIDE workflow
                logger.info("Web search failed or returned no models. Skipping MLE-STAR phases, using standard AIDE workflow")
                self.current_phase = "improve"
            else:
                logger.info(f"Web search successful, retrieved {len(models)} models. Moving to foundation phase.")
                self.current_phase = "foundation"
        
        # Phase 2: Foundation Building (generate initial solutions)
        if self.current_phase == "foundation":
            if len(self.initial_solutions) < len(self.retrieved_models) and self.retrieved_models:
                logger.info(f"Phase 2: Foundation Building ({len(self.initial_solutions)}/{len(self.retrieved_models)})")
                model_info = self.retrieved_models[len(self.initial_solutions)]
                solution = self._generate_initial_solution(model_info)
                if solution:
                    exec_result = exec_callback(solution.code)
                    self.parse_exec_result(solution, exec_result)
                    self.journal.append(solution)
                    self.initial_solutions.append(solution)
                else:
                    # Skip this model if generation failed
                    self.initial_solutions.append(None)
            else:
                # Merge solutions (Algorithm 1 from MLE-STAR paper)
                if len(self.initial_solutions) > 1:
                    logger.info("Phase 2: Merging solutions")
                    base = self.initial_solutions[0]
                    best_score = base.metric.value if base.metric else 0
                    
                    for ref_sol in self.initial_solutions[1:]:
                        merged = self._merge_solutions(base, ref_sol)
                        if merged:
                            exec_result = exec_callback(merged.code)
                            self.parse_exec_result(merged, exec_result)
                            self.journal.append(merged)
                            
                            new_score = merged.metric.value if merged.metric else 0
                            if (self.higher_is_better and new_score >= best_score) or \
                               (not self.higher_is_better and new_score <= best_score):
                                base = merged
                                best_score = new_score
                            else:
                                break
                
                self.current_phase = "refinement"
        
        # Phase 3: Targeted Refinement (ablation + refinement cycles)
        elif self.current_phase == "refinement":
            best_node = self.journal.get_best_node(only_good=True)
            if not best_node:
                best_node = self.journal.get_best_node(only_good=False)
            
            if best_node and len(self.ablation_history) < self.mlestar_cfg.refinement_iterations:
                logger.info(f"Phase 3: Targeted Refinement ({len(self.ablation_history)}/{self.mlestar_cfg.refinement_iterations})")
                
                # Ablation study
                ablation_node = self._ablation_study(best_node, self.ablation_history)
                if ablation_node:
                    exec_result = exec_callback(ablation_node.code)
                    self.parse_exec_result(ablation_node, exec_result)
                    self.journal.append(ablation_node)
                    
                    # Summarize ablation
                    ablation_summary = self._summarize_ablation(
                        ablation_node.code,
                        str(exec_result.get('feedback', ''))
                    )
                    self.ablation_history.append(ablation_summary)
                    
                    # Extract refinement plan
                    refine_plan = self._extract_refinement_plan(best_node, ablation_summary, self.refined_code_blocks)
                    if refine_plan:
                        # Refine the code block
                        refined_code = self._refine_code_block(
                            refine_plan['code_block'],
                            refine_plan['plan']
                        )
                        
                        if refined_code:
                            # Replace code block in solution
                            new_code = best_node.code.replace(
                                refine_plan['code_block'],
                                refined_code
                            )
                            
                            refined_node = Node(
                                plan=refine_plan['plan'],
                                code=new_code,
                                parent=best_node,
                                instruction_prompt=f"Refinement based on: {refine_plan['plan']}",
                                node_type="refined",
                                assistant=""
                            )
                            
                            exec_result = exec_callback(refined_node.code)
                            self.parse_exec_result(refined_node, exec_result)
                            self.journal.append(refined_node)
                            
                            self.refined_code_blocks.append({
                                'code_block': refine_plan['code_block'],
                                'plan': refine_plan['plan'],
                                'score': refined_node.metric.value if refined_node.metric else 0
                            })
            else:
                self.current_phase = "ensemble"
        
        # Phase 4: Ensemble
        elif self.current_phase == "ensemble":
            logger.info("Phase 4: Ensemble Creation")
            good_solutions = [n for n in self.journal.good_nodes if n.node_type in ["initial_solution", "merged_solution", "refined"]]
            
            if len(good_solutions) >= 2:
                ensemble_plan = self._suggest_ensemble_plan(good_solutions, self.ensemble_plans)
                if ensemble_plan:
                    ensemble_node = self._ensemble_solutions(good_solutions, ensemble_plan)
                    if ensemble_node:
                        exec_result = exec_callback(ensemble_node.code)
                        self.parse_exec_result(ensemble_node, exec_result)
                        self.journal.append(ensemble_node)
                        
                        self.ensemble_plans.append({
                            'plan': ensemble_plan,
                            'score': ensemble_node.metric.value if ensemble_node.metric else 0
                        })
            
            self.current_phase = "validation"
        
        # Phase 5: Validation (data leakage check, data usage check)
        elif self.current_phase == "validation":
            logger.info("Phase 5: Validation & Debugging")
            best_node = self.journal.get_best_node(only_good=True)
            if not best_node:
                best_node = self.journal.get_best_node(only_good=False)
            
            if best_node:
                # Check data leakage
                leakage_check = self._check_data_leakage(best_node.code)
                if leakage_check.get('leakage_status') == "Yes Data Leakage":
                    fixed_code = self._fix_data_leakage(best_node.code)
                    if fixed_code:
                        fixed_node = Node(
                            plan="Fixed data leakage",
                            code=fixed_code,
                            parent=best_node,
                            instruction_prompt="Data leakage fix",
                            node_type="leakage_fix",
                            assistant=""
                        )
                        exec_result = exec_callback(fixed_node.code)
                        self.parse_exec_result(fixed_node, exec_result)
                        self.journal.append(fixed_node)
                
                # Check data usage
                improved_code = self._check_data_usage(best_node.code, self.task_desc)
                if improved_code:
                    improved_node = Node(
                        plan="Incorporated unused information",
                        code=improved_code,
                        parent=best_node,
                        instruction_prompt="Data usage improvement",
                        node_type="data_usage_fix",
                        assistant=""
                    )
                    exec_result = exec_callback(improved_node.code)
                    self.parse_exec_result(improved_node, exec_result)
                    self.journal.append(improved_node)
            
            # After validation, fall back to standard AIDE improvement
            self.current_phase = "improve"
        
        # Fallback to standard AIDE behavior
        else:
            parent_node = self.search_policy()
            if parent_node is None:
                logger.info("Drafting...")
                result_node = self._draft()
            elif parent_node.is_buggy:
                logger.info("Debugging...")
                result_node = self._debug(parent_node)
            else:
                logger.info("Improving...")
                result_node = self._improve(parent_node)
            
            self.parse_exec_result(node=result_node, exec_result=exec_callback(result_node.code))
            self.journal.append(result_node)

