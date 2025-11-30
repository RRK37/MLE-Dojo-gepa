"""
MLE-STAR Agent: Machine Learning Engineering via Search and Targeted Refinement
Extends AIDE with MLE-STAR methodology including web search, HPO, refinement, and ablations.
"""
import logging
import random
import humanize
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

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
from mledojo.agent.mlestar.perplexity_client import PerplexityClient

logger = logging.getLogger("mlestar")

ExecCallbackType = Callable[[str], Dict]


@dataclass
class LLMConfig:
    """Configuration for LLM API settings"""
    model_mode: str
    model_name: str
    port: int = 8314
    max_completion_tokens: int = 8192
    max_prompt_tokens: int = 30000
    api_idx: int = -1
    api_key: str = None
    temperature: float = 0.0
    top_p: float = 1.0


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
        search_iterations: int = 3,
        refinement_iterations: int = 5,
        perplexity_api_key: Optional[str] = None,
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
            search_iterations: Number of web search iterations
            refinement_iterations: Number of refinement cycles
            perplexity_api_key: Perplexity API key (optional, can use env var)
        """
        # Initialize parent AIDE agent
        super().__init__(task_desc, cfg, journal, higher_is_better, data_dir, output_dir)
        
        # MLE-STAR specific configuration
        self.search_iterations = search_iterations
        self.refinement_iterations = refinement_iterations
        
        # Initialize Perplexity client
        try:
            self.perplexity_client = PerplexityClient(api_key=perplexity_api_key)
            self.perplexity_enabled = True
        except Exception as e:
            logger.warning(f"Perplexity not available: {e}. Web search will be disabled.")
            self.perplexity_client = None
            self.perplexity_enabled = False
        
        # MLE-STAR state tracking
        self.retrieved_models: List[Dict[str, str]] = []
        self.initial_solutions: List[Node] = []
        self.ablation_results: List[Dict] = []
        self.refined_code_blocks: List[Dict] = []
        self.ensemble_plans: List[Dict] = []
        self.current_phase = "search"  # search, foundation, refinement, ensemble, validation
        
    def _web_search_phase(self) -> List[Dict[str, str]]:
        """
        Phase 1: Web search for models (MLE-STAR Prompt 1).
        Uses Perplexity to search for effective models.
        """
        if not self.perplexity_enabled:
            logger.warning("Perplexity not available, skipping web search phase")
            return []
        
        logger.info(f"Starting web search phase ({self.search_iterations} iterations)")
        all_models = []
        
        for i in range(self.search_iterations):
            logger.info(f"Web search iteration {i+1}/{self.search_iterations}")
            models = self.perplexity_client.search_models(self.task_desc, num_models=5)
            all_models.extend(models)
        
        # Deduplicate models by name
        seen_names = set()
        unique_models = []
        for model in all_models:
            if model['model_name'] not in seen_names:
                seen_names.add(model['model_name'])
                unique_models.append(model)
        
        self.retrieved_models = unique_models[:10]  # Keep top 10
        logger.info(f"Retrieved {len(self.retrieved_models)} unique models")
        return self.retrieved_models
    
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
            self._web_search_phase()
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
            
            if best_node and len(self.ablation_results) < self.refinement_iterations:
                logger.info(f"Phase 3: Targeted Refinement ({len(self.ablation_results)}/{self.refinement_iterations})")
                
                # Ablation study
                ablation_node = self._ablation_study(best_node, self.ablation_results)
                if ablation_node:
                    exec_result = exec_callback(ablation_node.code)
                    self.parse_exec_result(ablation_node, exec_result)
                    self.journal.append(ablation_node)
                    
                    # Summarize ablation
                    ablation_summary = self._summarize_ablation(
                        ablation_node.code,
                        str(exec_result.get('feedback', ''))
                    )
                    self.ablation_results.append(ablation_summary)
                    
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

