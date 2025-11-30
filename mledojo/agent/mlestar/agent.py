"""
MLE-STAR: Machine Learning Engineering - Search, Target, and Refine

This module implements the MLE-STAR agent following the exact methodology
from PaperMethod.md (Algorithms 1, 2, 3) with MLE-Dojo integration.
"""

import os
import logging
import json
import re
import tiktoken
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

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
from .prompt import prompts
from .config import MLEStarConfig

logger = logging.getLogger("mlestar")

ExecCallbackType = Callable[[str], Dict]


class MLEStarAgent:
    """
    MLE-STAR Agent implementing Algorithms 1, 2, 3 from PaperMethod.md
    
    Algorithm 1: Search → Initial Solution (merging)
    Algorithm 2: Targeted Refinement (ablation → extract → refine)
    Algorithm 3: Ensemble Strategies
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
        self.initial_solutions: List[Tuple[str, float]] = []  # (code, score)
        self.best_solution: Optional[str] = None
        self.best_score: float = -float('inf') if higher_is_better else float('inf')
        self.ablation_history: List[str] = []
        self.refined_code_blocks: List[str] = []
        self.refinement_plans: List[Tuple[str, float]] = []  # (plan, score)
        self.ensemble_solutions: List[str] = []
        self.ensemble_plans: List[Tuple[str, float]] = []
        
        # Tracking for graphs and errors
        self.reward_history: List[Dict] = []  # [{"timestep": int, "reward": float, "phase": str, "iteration": int, "error": Optional[str]}]
        self.error_history: List[Dict] = []  # [{"timestep": int, "phase": str, "error": str, "traceback": str}]
        self.timestep = 0
        
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
    
    def _init_web_client(self):
        """Initialize web search client (Perplexity)."""
        self.web_client = None
        # Support both PPLX_API_KEY (Alxandria) and PERPLEXITY_API_KEY (MLE-STAR)
        perplexity_key = os.getenv('PPLX_API_KEY') or os.getenv('PERPLEXITY_API_KEY')
        if perplexity_key:
            try:
                self.web_client = OpenAI(
                    api_key=perplexity_key,
                    base_url="https://api.perplexity.ai"
                )
                logger.info("Initialized Perplexity web client")
            except Exception as e:
                logger.error(f"Failed to initialize Perplexity client: {str(e)}")
    
    def query_llm(self, system_message: str, user_message: Optional[str] = None) -> Tuple[str, float]:
        """Query the LLM model."""
        system_message = compile_prompt_to_md(system_message) if system_message else None
        user_message = compile_prompt_to_md(user_message) if user_message else None
        messages = opt_messages_to_list(system_message, user_message)
        
        response = self.model_client.chat_completion(messages, self.model_settings)
        cost = 0.0  # TODO: Calculate actual cost
        self.total_cost += cost
        return response, cost
    
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
    
    def parse_json_response(self, text: str) -> List[Dict]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Try to find JSON in the text
        try:
            return json.loads(text)
        except:
            # Try to extract JSON-like structures
            matches = re.findall(r'\{[^{}]*"model_name"[^{}]*"example_code"[^{}]*\}', text)
            results = []
            for match in matches:
                try:
                    results.append(json.loads(match))
                except:
                    pass
            return results
            
    # ========== Algorithm 1: Search → Initial Solution ==========
    
    def algorithm_1_initial_solution(self, exec_callback: ExecCallbackType) -> str:
        """
        Algorithm 1: Generating an initial solution
        
        Input: task description T_task, datasets D, score function h, number of retrieved models M
        1. {T_model_i, T_code_i}_{i=1..M} = A_retriever(T_task)
        2-5. for i = 1 to M: generate and evaluate s_init_i
        6-17. Sequential merging: s_0 ← best, then merge others until no improvement
        """
        M = self.mlestar_cfg.num_models_to_retrieve
        
        # Step 1: Retrieve models (A_retriever)
        logger.info(f"Step 1: Retrieving {M} models...")
        self.retrieved_models = self._retrieve_models(M)
        
        if not self.retrieved_models:
            logger.warning("No models retrieved, generating default solution")
            return self._generate_default_solution(exec_callback)
        
        # Steps 2-5: Generate and evaluate initial solutions
        logger.info(f"Step 2-5: Generating and evaluating {len(self.retrieved_models)} solutions...")
        for i, model in enumerate(self.retrieved_models):
            logger.info(f"Generating solution {i+1}/{len(self.retrieved_models)}")
            code = self._generate_initial_solution_from_model(model)
            if code:
                result = exec_callback(code)
                score = self._extract_score(result)
                self.initial_solutions.append((code, score))
                logger.info(f"Solution {i+1} score: {score}")
        
        if not self.initial_solutions:
            return self._generate_default_solution(exec_callback)
        
        # Steps 6-17: Sequential merging
        logger.info("Step 6-17: Sequential merging...")
        # Sort by score (best first)
        self.initial_solutions.sort(key=lambda x: x[1], reverse=self.higher_is_better)
        s_0, h_best = self.initial_solutions[0]
        self.best_solution = s_0
        self.best_score = h_best
        
        # Try merging others (Algorithm 1, lines 8-17)
        for i in range(1, len(self.initial_solutions)):
            s_candidate_code, s_candidate_score = self.initial_solutions[i]
            merged_code = self._merge_solutions(s_0, s_candidate_code)
            if not merged_code:
                logger.info(f"Failed to merge solution {i+1}, skipping")
                continue
            
            result = exec_callback(merged_code)
            h_candidate = self._extract_score(result)
            
            # Algorithm 1, line 11: if h(s_candidate) ≥ h_best then
            if (self.higher_is_better and h_candidate >= h_best) or \
               (not self.higher_is_better and h_candidate <= h_best):
                s_0 = merged_code
                h_best = h_candidate
                self.best_solution = s_0
                self.best_score = h_best
                logger.info(f"Merged solution {i+1}, new score: {h_best}")
            else:
                # Algorithm 1, line 14-15: else break
                logger.info(f"Merged solution {i+1} did not improve (score: {h_candidate} vs best: {h_best}), stopping merge")
                break
        
        return self.best_solution
    
    def _retrieve_models(self, M: int) -> List[Dict]:
        """A_retriever: Retrieve M models via web search (Prompt 1)."""
        query = f"Kaggle competition winning solutions for: {self.task_desc[:200]}"
        search_result = self.search_web(query)  # Now synchronous
        
        prompt = prompts.prompt_1_model_retrieval(self.task_desc, M)
        response, _ = self.query_llm(prompt)
        
        models = self.parse_json_response(response)
        if not models:
            # Fallback: create dummy models
            models = [
                {"model_name": f"Model_{i+1}", "example_code": "# Default model code"}
                for i in range(M)
            ]
        
        return models[:M]
    
    def _generate_initial_solution_from_model(self, model: Dict) -> Optional[str]:
        """A_init: Generate initial solution from model (Prompt 2)."""
        model_desc = model.get("model_name", "")
        example_code = model.get("example_code", "")
        
        prompt = prompts.prompt_2_initial_solution(
            self.task_desc,
            model_desc,
            example_code
        )
        
        response, _ = self.query_llm(prompt)
        code = extract_code(response)
        return code if code else None
    
    def _merge_solutions(self, base_code: str, reference_code: str) -> Optional[str]:
        """A_merger: Merge base and reference solutions (Prompt 3)."""
        prompt = prompts.prompt_3_merge_solutions(
            self.task_desc,
            base_code,
            reference_code
        )
        
        response, _ = self.query_llm(prompt)
        code = extract_code(response)
        return code if code else None
    
    def _generate_default_solution(self, exec_callback: ExecCallbackType) -> str:
        """Generate a default solution if retrieval fails."""
        prompt = prompts.prompt_2_initial_solution(
            self.task_desc,
            "Default baseline model",
            "# Simple baseline implementation"
        )
        response, _ = self.query_llm(prompt)
        code = extract_code(response)
        if code:
            result = exec_callback(code)
            score = self._extract_score(result)
            self.best_solution = code
            self.best_score = score
            return code
        return "# Default solution placeholder"
    
    # ========== Algorithm 2: Targeted Refinement ==========
    
    def algorithm_2_refinement(self, s_0: str, exec_callback: ExecCallbackType) -> str:
        """
        Algorithm 2: Refining solution
        
        Input: initial solution s_0, outer loop steps T, inner loop steps K
        1-3. Initialize
        4-28. for t = 0 to T-1:
            - Ablation study (A_abl)
            - Summarize (A_summarize)
            - Extract code block + plan (A_extractor)
            - Refine code block (A_coder) with K inner iterations
        """
        s_final = s_0
        h_best = self.best_score
        T = self.mlestar_cfg.refinement_iterations
        K = self.mlestar_cfg.inner_refinement_steps
        T_abl = []  # Ablation summaries
        C = []  # Refined code blocks
        
        for t in range(T):
            logger.info(f"Refinement iteration {t+1}/{T}")
            s_t = s_final
            
            # Step 5: Ablation study
            ablation_code = self._generate_ablation_study(s_t, T_abl)
            if ablation_code:
                ablation_result = exec_callback(ablation_code)
                ablation_output = ablation_result.get("feedback", {}).get("stdout", "")
                
                # Step 7: Summarize ablation
                ablation_summary = self._summarize_ablation(ablation_code, ablation_output)
                T_abl.append(ablation_summary)
            
            # Step 8: Extract code block + plan
            if T_abl:
                code_block, plan = self._extract_refine_plan(s_t, T_abl[-1], C)
                if not code_block or not plan:
                    continue
            else:
                continue
            
            # Step 9: First refinement (k=0)
            c_t_0 = self._refine_code_block(code_block, plan)
            s_t_0 = s_t.replace(code_block, c_t_0)
            result = exec_callback(s_t_0)
            h_t_0 = self._extract_score(result)
            
            if (self.higher_is_better and h_t_0 >= h_best) or \
               (not self.higher_is_better and h_t_0 <= h_best):
                s_final = s_t_0
                h_best = h_t_0
                self.best_solution = s_final
                self.best_score = h_best
            
            # Steps 16-25: Inner loop (k = 1 to K-1)
            for k in range(1, K):
                # Step 17: Alternative plan
                alt_plan = self._suggest_alternative_plan(code_block, self.refinement_plans)
                if not alt_plan:
                    break
                
                # Step 18: Refine with alternative plan
                c_t_k = self._refine_code_block(code_block, alt_plan)
                s_t_k = s_t.replace(code_block, c_t_k)
                result = exec_callback(s_t_k)
                h_t_k = self._extract_score(result)
                
                if (self.higher_is_better and h_t_k >= h_best) or \
                   (not self.higher_is_better and h_t_k <= h_best):
                    s_final = s_t_k
                    h_best = h_t_k
                    self.best_solution = s_final
                    self.best_score = h_best
                
                self.refinement_plans.append((alt_plan, h_t_k))
            
            # Steps 26-27: Update history
            C.append(code_block)
        
        return s_final
    
    def _generate_ablation_study(self, solution: str, previous_ablations: List[str]) -> Optional[str]:
        """A_abl: Generate ablation study (Prompt 4)."""
        prompt = prompts.prompt_4_ablation_study(solution, previous_ablations)
        response, _ = self.query_llm(prompt)
        code = extract_code(response)
        return code if code else None
    
    def _summarize_ablation(self, ablation_code: str, raw_result: str) -> str:
        """A_summarize: Summarize ablation results (Prompt 5)."""
        prompt = prompts.prompt_5_summarize_ablation(ablation_code, raw_result)
        response, _ = self.query_llm(prompt)
        return response
    
    def _extract_refine_plan(self, solution: str, ablation_summary: str, prev_blocks: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """A_extractor: Extract code block and refinement plan (Prompt 6)."""
        prompt = prompts.prompt_6_extract_refine_plan(solution, ablation_summary, prev_blocks)
        response, _ = self.query_llm(prompt)
        
        # Parse JSON response
        plans = self.parse_json_response(response)
        if plans and len(plans) > 0:
            plan_data = plans[0]
            code_block = plan_data.get("code_block", "")
            plan = plan_data.get("plan", "")
            return code_block, plan
        
        # Fallback: extract code block from response
        code_block = extract_code(response)
        plan = extract_text_up_to_code(response)
        return code_block if code_block else None, plan if plan else None
    
    def _refine_code_block(self, code_block: str, plan: str) -> str:
        """A_coder: Refine code block (Prompt 7)."""
        prompt = prompts.prompt_7_refine_code_block(code_block, plan)
        response, _ = self.query_llm(prompt)
        refined = extract_code(response)
        return refined if refined else code_block
    
    def _suggest_alternative_plan(self, code_block: str, prev_plans: List[Tuple[str, float]]) -> Optional[str]:
        """A_planner: Suggest alternative refinement plan (Prompt 8)."""
        if not prev_plans:
            return None
        
        plans = [p[0] for p in prev_plans]
        scores = [p[1] for p in prev_plans]
        
        prompt = prompts.prompt_8_alternative_plan(code_block, plans, scores)
        response, _ = self.query_llm(prompt)
        plan = extract_text_up_to_code(response)
        return plan if plan else None
    
    # ========== Algorithm 3: Ensemble ==========
    
    def algorithm_3_ensemble(self, solutions: List[str], exec_callback: ExecCallbackType) -> str:
        """
        Algorithm 3: Ensembling final solutions
        
        Input: candidate final solutions s_final^1, ..., s_final^L, ensemble loop steps R
        1-3. Initial ensemble plan and evaluation
        4-8. for r = 1 to R-1: alternative ensemble plans
        9-11. Return best ensemble
        """
        L = len(solutions)
        R = self.mlestar_cfg.ensemble_iterations
        
        if L < 2:
            return solutions[0] if solutions else ""
        
        self.ensemble_solutions = solutions
        
        # Step 1: Initial ensemble plan
        ensemble_plan = self._suggest_ensemble_plan(solutions, [])
        if not ensemble_plan:
            return solutions[0]
        
        # Step 2: Implement ensemble
        s_ens_0 = self._implement_ensemble(solutions, ensemble_plan)
        result = exec_callback(s_ens_0)
        h_ens_0 = self._extract_score(result)
        
        # Track all ensembles for argmax selection (Algorithm 3, line 9)
        ensemble_results = [(s_ens_0, h_ens_0, 0)]  # (solution, score, iteration)
        self.ensemble_plans.append((ensemble_plan, h_ens_0))
        
        # Steps 4-8: Alternative ensemble plans (Algorithm 3, lines 4-8)
        for r in range(1, R):
            ensemble_plan_r = self._suggest_ensemble_plan(solutions, self.ensemble_plans)
            if not ensemble_plan_r:
                break
            
            s_ens_r = self._implement_ensemble(solutions, ensemble_plan_r)
            result = exec_callback(s_ens_r)
            h_ens_r = self._extract_score(result)
            
            ensemble_results.append((s_ens_r, h_ens_r, r))
            self.ensemble_plans.append((ensemble_plan_r, h_ens_r))
        
        # Algorithm 3, lines 9-10: r* = argmax, s_ens* = s_ens^{r*}
        if self.higher_is_better:
            best_idx = max(range(len(ensemble_results)), key=lambda i: ensemble_results[i][1])
        else:
            best_idx = min(range(len(ensemble_results)), key=lambda i: ensemble_results[i][1])
        
        best_ensemble, best_score, best_r = ensemble_results[best_idx]
        logger.info(f"Best ensemble from iteration {best_r} with score: {best_score}")
        return best_ensemble
    
    def _suggest_ensemble_plan(self, solutions: List[str], prev_plans: List[Tuple[str, float]]) -> Optional[str]:
        """A_ens_planner: Suggest ensemble plan (Prompt 9)."""
        plans = [p[0] for p in prev_plans]
        scores = [p[1] for p in prev_plans]
        
        prompt = prompts.prompt_9_ensemble_plan(solutions, plans, scores)
        response, _ = self.query_llm(prompt)
        plan = extract_text_up_to_code(response)
        return plan if plan else None
    
    def _implement_ensemble(self, solutions: List[str], plan: str) -> str:
        """A_ensembler: Implement ensemble (Prompt 10)."""
        prompt = prompts.prompt_10_implement_ensemble(solutions, plan)
        response, _ = self.query_llm(prompt)
        code = extract_code(response)
        return code if code else solutions[0]
    
    # ========== Validation & Debugging ==========
    
    def _debug_code(self, code: str, error: str) -> Optional[str]:
        """Debug code with error (Prompt 11)."""
        prompt = prompts.prompt_11_debug(code, error)
        response, _ = self.query_llm(prompt)
        fixed_code = extract_code(response)
        return fixed_code if fixed_code else None
    
    def _check_data_leakage(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check for data leakage (Prompt 12)."""
        prompt = prompts.prompt_12_check_leakage(code)
        response, _ = self.query_llm(prompt)
        
        answers = self.parse_json_response(response)
        if answers and len(answers) > 0:
            answer = answers[0]
            leakage_status = answer.get("leakage_status", "").lower()
            code_block = answer.get("code_block", "")
            has_leakage = "yes" in leakage_status
            return has_leakage, code_block
        return False, None
    
    def _fix_data_leakage(self, code: str) -> Optional[str]:
        """Fix data leakage (Prompt 13)."""
        prompt = prompts.prompt_13_fix_leakage(code)
        response, _ = self.query_llm(prompt)
        fixed_code = extract_code(response)
        return fixed_code if fixed_code else None
    
    def _check_data_usage(self, solution: str) -> Optional[str]:
        """Check if all data is used (Prompt 14)."""
        prompt = prompts.prompt_14_check_data_usage(solution, self.task_desc)
        response, _ = self.query_llm(prompt)
        
        if "All the provided information is used" in response:
            return None
        
        improved_code = extract_code(response)
        return improved_code if improved_code else None
    
    # ========== Utility Methods ==========
    
    def _extract_score(self, exec_result: Dict) -> float:
        """Extract score from execution result."""
        if exec_result.get("action_status") == "FAILED":
            return -float('inf') if self.higher_is_better else float('inf')
        
        position_score = exec_result.get("current_position_score", 0.0)
        return float(position_score)
    
    def update_data_preview(self):
        """Update data preview."""
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)
    
    def step(self, exec_callback: ExecCallbackType):
        """
        Main step function - executes one phase of MLE-STAR workflow.
        
        This follows the MLE-STAR methodology:
        1. Algorithm 1: Search → Initial Solution
        2. Algorithm 2: Targeted Refinement
        3. Algorithm 3: Ensemble
        4. Validation & Debugging
        """
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        
        # Determine current phase
        if not hasattr(self, '_phase'):
            self._phase = 'initial'
        
        # Optional print statements (like AIDE) - prints are always shown, just like AIDE
        # AIDE doesn't have verbose flag, prints are always shown
        if self._phase == 'initial':
            print("Phase: Algorithm 1 - Initial Solution (Search → Merge)")
        elif self._phase == 'refinement':
            print("Phase: Algorithm 2 - Targeted Refinement (Ablation → Refine)")
        elif self._phase == 'ensemble':
            print("Phase: Algorithm 3 - Ensemble")
        elif self._phase == 'validation':
            print("Phase: Validation & Debugging")
        elif self._phase == 'done':
            print("MLE-STAR workflow completed")
        
        if self._phase == 'initial':
            logger.info("Phase: Algorithm 1 - Initial Solution")
            try:
                solution = self.algorithm_1_initial_solution(exec_callback)
                node = Node(
                    code=solution,
                    plan="Initial solution from Algorithm 1",
                    node_type="draft",
                    instruction_prompt="Algorithm 1: Search → Initial Solution"
                )
                result = exec_callback(solution)
                reward = self._extract_score(result)
                self._record_reward(reward, 'initial', iteration=0)
                self.parse_exec_result(node, result)
                self.journal.append(node)
                self._phase = 'refinement'
            except Exception as e:
                self._record_error('initial', str(e))
                logger.error(f"Error in initial phase: {str(e)}", exc_info=True)
                raise
            
        elif self._phase == 'refinement':
            logger.info("Phase: Algorithm 2 - Targeted Refinement")
            if self.best_solution:
                try:
                    refinement_iter = len(self.ablation_history)
                    solution = self.algorithm_2_refinement(self.best_solution, exec_callback)
                    node = Node(
                        code=solution,
                        plan="Refined solution from Algorithm 2",
                        node_type="improve",
                        instruction_prompt="Algorithm 2: Targeted Refinement"
                    )
                    result = exec_callback(solution)
                    reward = self._extract_score(result)
                    self._record_reward(reward, 'refinement', iteration=refinement_iter)
                    self.parse_exec_result(node, result)
                    self.journal.append(node)
                except Exception as e:
                    self._record_error('refinement', str(e))
                    logger.error(f"Error in refinement phase: {str(e)}", exc_info=True)
                    raise
            self._phase = 'ensemble'
            
        elif self._phase == 'ensemble':
            logger.info("Phase: Algorithm 3 - Ensemble")
            solutions = [s[0] for s in self.initial_solutions[:3]]  # Top 3 solutions
            if self.best_solution:
                solutions.append(self.best_solution)
            
            if len(solutions) >= 2:
                try:
                    ensemble_iter = len(self.ensemble_plans)
                    solution = self.algorithm_3_ensemble(solutions, exec_callback)
                    node = Node(
                        code=solution,
                        plan="Ensemble solution from Algorithm 3",
                        node_type="improve",
                        instruction_prompt="Algorithm 3: Ensemble"
                    )
                    result = exec_callback(solution)
                    reward = self._extract_score(result)
                    self._record_reward(reward, 'ensemble', iteration=ensemble_iter)
                    self.parse_exec_result(node, result)
                    self.journal.append(node)
                except Exception as e:
                    self._record_error('ensemble', str(e))
                    logger.error(f"Error in ensemble phase: {str(e)}", exc_info=True)
                    raise
            self._phase = 'validation'
            
        elif self._phase == 'validation':
            logger.info("Phase: Validation & Debugging")
            if self.best_solution:
                try:
                    # Check for data leakage
                    has_leakage, _ = self._check_data_leakage(self.best_solution)
                    if has_leakage:
                        fixed = self._fix_data_leakage(self.best_solution)
                        if fixed:
                            self.best_solution = fixed
                    
                    # Check data usage
                    improved = self._check_data_usage(self.best_solution)
                    if improved:
                        self.best_solution = improved
                    
                    # Record final reward
                    if self.best_solution:
                        result = exec_callback(self.best_solution)
                        reward = self._extract_score(result)
                        self._record_reward(reward, 'validation', iteration=0)
                except Exception as e:
                    self._record_error('validation', str(e))
                    logger.error(f"Error in validation phase: {str(e)}", exc_info=True)
            self._phase = 'done'
        
        else:
            logger.info("MLE-STAR workflow completed")
    
    def _record_reward(self, reward: float, phase: str, iteration: int):
        """Record reward for graphing."""
        self.timestep += 1
        self.reward_history.append({
            "timestep": self.timestep,
            "reward": reward,
            "phase": phase,
            "iteration": iteration,
            "error": None
        })
    
    def _record_error(self, phase: str, error: str, traceback_str: Optional[str] = None):
        """Record error for tracking."""
        import traceback as tb
        if traceback_str is None:
            traceback_str = ''.join(tb.format_exc())
        
        self.timestep += 1
        self.error_history.append({
            "timestep": self.timestep,
            "phase": phase,
            "error": error,
            "traceback": traceback_str
        })
        
        # Also record in reward history with error flag
        self.reward_history.append({
            "timestep": self.timestep,
            "reward": -float('inf') if self.higher_is_better else float('inf'),
            "phase": phase,
            "iteration": 0,
            "error": error
        })
    
    def save_graphs(self, output_dir: str):
        """Save reward graphs with phase and iteration information."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from pathlib import Path
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if not self.reward_history:
                logger.warning("No reward history to plot")
                return
            
            # Prepare data
            timesteps = [r["timestep"] for r in self.reward_history]
            rewards = [r["reward"] for r in self.reward_history]
            phases = [r["phase"] for r in self.reward_history]
            iterations = [r["iteration"] for r in self.reward_history]
            has_errors = [r["error"] is not None for r in self.reward_history]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Reward over timestep with phase colors
            phase_colors = {
                'initial': '#e3f2fd',
                'refinement': '#fff3e0',
                'ensemble': '#f3e5f5',
                'validation': '#e0f2f1',
                'done': '#ffebee'
            }
            
            # Plot rewards by phase
            for phase in set(phases):
                phase_mask = [p == phase for p in phases]
                phase_timesteps = [t for t, m in zip(timesteps, phase_mask) if m]
                phase_rewards = [r for r, m in zip(rewards, phase_mask) if m]
                ax1.scatter(phase_timesteps, phase_rewards, 
                           c=phase_colors.get(phase, '#cccccc'), 
                           label=phase, s=100, alpha=0.7, edgecolors='black', linewidths=1)
            
            # Plot line connecting rewards
            ax1.plot(timesteps, rewards, 'k-', alpha=0.3, linewidth=1)
            
            # Mark errors
            error_timesteps = [t for t, e in zip(timesteps, has_errors) if e]
            error_rewards = [r for r, e in zip(rewards, has_errors) if e]
            if error_timesteps:
                ax1.scatter(error_timesteps, error_rewards, 
                           c='red', marker='x', s=200, linewidths=3, 
                           label='Error', zorder=10)
            
            ax1.set_xlabel('Timestep', fontsize=12)
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.set_title('MLE-STAR: Reward per Timestep by Phase', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Reward by iteration within each phase
            phase_iter_data = {}
            for r in self.reward_history:
                phase = r["phase"]
                iteration = r["iteration"]
                reward = r["reward"]
                if phase not in phase_iter_data:
                    phase_iter_data[phase] = {}
                if iteration not in phase_iter_data[phase]:
                    phase_iter_data[phase][iteration] = []
                phase_iter_data[phase][iteration].append(reward)
            
            # Plot bars for each phase-iteration combination
            x_pos = 0
            x_labels = []
            x_positions = []
            bar_colors = []
            bar_heights = []
            
            for phase in ['initial', 'refinement', 'ensemble', 'validation']:
                if phase in phase_iter_data:
                    for iteration in sorted(phase_iter_data[phase].keys()):
                        avg_reward = sum(phase_iter_data[phase][iteration]) / len(phase_iter_data[phase][iteration])
                        x_labels.append(f"{phase}\niter{iteration}")
                        x_positions.append(x_pos)
                        bar_heights.append(avg_reward)
                        bar_colors.append(phase_colors.get(phase, '#cccccc'))
                        x_pos += 1
            
            if x_positions:
                bars = ax2.bar(x_positions, bar_heights, color=bar_colors, 
                              edgecolor='black', linewidth=1, alpha=0.7)
                ax2.set_xticks(x_positions)
                ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
                ax2.set_ylabel('Average Reward', fontsize=12)
                ax2.set_title('MLE-STAR: Average Reward by Phase and Iteration', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save figure
            graph_path = output_path / "mlestar_rewards.png"
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved reward graph to {graph_path}")
            
            # Save error log if there are errors
            if self.error_history:
                error_path = output_path / "mlestar_errors.json"
                import json
                with open(error_path, 'w') as f:
                    json.dump(self.error_history, f, indent=2)
                logger.info(f"Saved error log to {error_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping graph generation")
        except Exception as e:
            logger.error(f"Failed to save graphs: {str(e)}", exc_info=True)
    
    def parse_exec_result(self, node: Node, exec_result: Dict):
        """Parse execution result and update node."""
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
            node.metric = WorstMetricValue(value=0.0)
        else:
            node.metric = MetricValue(eval_result.position_score, maximize=self.higher_is_better)
