from gepa import GEPAAdapter, EvaluationBatch
from typing import Callable, Any
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
import csv
import os
from datetime import datetime

class MLEDojoGEPAAdapter(GEPAAdapter):
    def __init__(self, 
                 competition_name: str,
                 data_dir: str,
                 output_dir: str,
                 agent_factory: Callable[[str], Any], 
                 max_steps: int = 10,
                 execution_timeout: int = 600,
                 score_mode: str = "position",
                 log_dir: str = "./logs"):
        """
        Args:
            competition_name: The Kaggle competition name (e.g., 'titanic').
            data_dir: Path to competition data directory.
            output_dir: Path to output directory for env results.
            agent_factory: A function that takes `system_prompt` and returns a FRESH Agent instance.
            max_steps: Safety limit for agent iterations per episode.
            execution_timeout: Timeout for code execution in seconds.
            score_mode: Scoring mode ('position' or 'raw').
            log_dir: Directory to save CV score logs.
        """
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.agent_factory = agent_factory
        self.score_mode = score_mode
        self.log_dir = log_dir
        
        # Initialize CV score logging
        os.makedirs(log_dir, exist_ok=True)
        self.cv_log_file = os.path.join(log_dir, f"cv_scores_{competition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_cv_log()
        
        # Track GEPA and agent iterations
        self.gepa_iteration = 0
        self.total_evaluations = 0
    
    def _init_cv_log(self):
        """Initialize the CSV file for logging CV scores."""
        with open(self.cv_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'gepa_iteration', 
                'episode_idx',
                'agent_step',
                'cv_score',
                'execution_status',
                'has_submission',
                'final_reward',
                'prompt_preview'
            ])
        print(f"[Adapter] Initialized CV score log: {self.cv_log_file}")
    
    def _log_cv_score(self, gepa_iter, episode_idx, agent_step, cv_score, exec_status, has_submission, final_reward, prompt_preview):
        """Log a CV score to the CSV file."""
        with open(self.cv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                gepa_iter,
                episode_idx,
                agent_step,
                cv_score,
                exec_status,
                has_submission,
                final_reward,
                prompt_preview
            ])

    def evaluate(self, batch: list, candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        self.log_dir = log_dir
        
        # Initialize CV score logging
        os.makedirs(log_dir, exist_ok=True)
        self.cv_log_file = os.path.join(log_dir, f"cv_scores_{competition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_cv_log()
        
        # Track GEPA and agent iterations
        self.gepa_iteration = 0
        self.total_evaluations = 0

    def evaluate(self, batch: list, candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """
        Runs the agent against the environment using the candidate prompt.
        """
        # 1. Get the new prompt from GEPA (candidate is a dict, not an object)
        base_system_prompt = candidate['system_prompt']
        
        # 2. Enhance the prompt to ALWAYS create submission files
        # This is critical for getting actual rewards from the environment
        system_prompt = f"""{base_system_prompt}

CRITICAL INSTRUCTIONS FOR EVERY CODE ITERATION:
1. Load the training data and train your model
2. Load the test data from the input directory
3. Make predictions on the test set using your trained model  
4. Create a submission DataFrame matching the format in sample_submission.csv
5. Save to 'submission.csv' in the output directory (NOT the input directory)
6. Your score depends ENTIRELY on creating this valid submission.csv file

Code Template Structure:
```python
import pandas as pd
# 1. Load and preprocess train data
train = pd.read_csv('/path/to/train.csv')
# ... feature engineering ...

# 2. Train model
model.fit(X_train, y_train)

# 3. Load test data and preprocess identically
test = pd.read_csv('/path/to/test.csv')
# ... same feature engineering ...

# 4. Make predictions
predictions = model.predict(X_test)

# 5. Create submission matching sample_submission.csv format
submission = pd.DataFrame({{'PassengerId': test['PassengerId'], 'Survived': predictions}})

# 6. Save to output directory
submission.to_csv('submission.csv', index=False)  # This creates it in output_dir
```

REMEMBER: You MUST create submission.csv in EVERY iteration. Without it, your score is 0."""
        
        print(f"\n[Adapter] Evaluating with base prompt: {base_system_prompt[:80]}...")
        print(f"[Adapter] Batch size: {len(batch)}, Capture traces: {capture_traces}")
        
        # 2. Run Episodes (1 episode per batch item)
        # Each batch item represents one independent run of the agent with this prompt
        num_episodes = len(batch) if batch else 1
        print(f"[Adapter] Running {num_episodes} episode(s) to evaluate this prompt")
        
        total_score = 0
        full_traces = []
        rollout_outputs = []

        for episode_idx in range(num_episodes):
            # Create a fresh environment and agent for this run
            # Setup competition registry
            competition_registry = CompetitionRegistry()
            comp_info = CompInfo()
            metric_class = get_metric(self.competition_name)
            
            competition_registry.register(
                self.competition_name,
                data_dir=self.data_dir,
                comp_info=comp_info,
                metric_class=metric_class
            )
            
            env = KaggleEnvironment(
                competition_name=self.competition_name,
                output_dir=self.output_dir,
                competition_registry=competition_registry,
                render_mode=None,
                execution_timeout=self.execution_timeout,
                score_mode=self.score_mode
            )
            agent = self.agent_factory(system_prompt)
            
            # Initialize the agent's data preview (required before first step)
            agent.update_data_preview()
            
            obs, _ = env.reset()
            done = False
            steps = 0
            
            print(f"[Adapter] Environment reset, starting agent loop (max {self.max_steps} steps)")
            
            # This list will collect the "Thought Process" for GEPA to analyze
            episode_trace = [] 

            # Define the callback that connects Agent.step() -> Gym.step()
            def env_bridge_callback(code_to_execute: str) -> dict:
                nonlocal obs, done, steps
                
                # Log what the agent tried to do
                episode_trace.append(f"\n--- Step {steps} ---")
                episode_trace.append(f"Generated Code:\n{code_to_execute[:500]}...") # Truncate for brevity in logs
                
                # Execute in Gym using the MLE-Dojo action API
                # MLE-Dojo envs use action="execute_code" with code=<code_string> as kwarg
                try:
                    result = env.step(action="execute_code", code=code_to_execute)
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                    elif len(result) == 2:
                        obs, reward_or_info = result
            
                        # Check if second value is a dict (info) or float (reward)
                        if isinstance(reward_or_info, dict):
                            info = reward_or_info
                            reward = info.get("reward", 0.0)
                        else:
                            reward = float(reward_or_info)
                            info = {}
                        terminated = False
                        truncated = False
                    else:
                        raise ValueError(f"Unexpected number of return values from env.step(): {len(result)}")
                except Exception as e:
                    print(f"[Adapter] Error in env.step(): {e}")
                    obs = {"status": "FAILED", "feedback": str(e)}
                    reward = 0.0
                    terminated = True
                    truncated = False
                    info = {}
                
                # Use CV score as reward if found and no submission reward exists
                has_submission = reward > 0.0
                if cv_score > 0 and reward == 0.0:
                    reward = cv_score
                    print(f"[Adapter] ✓ Using CV score from iteration output as reward: {reward:.4f}")
                
                # Log CV score if extracted
                if cv_score > 0:
                    self._log_cv_score(
                        gepa_iter=self.gepa_iteration,
                        episode_idx=episode_idx,
                        agent_step=steps,
                        cv_score=cv_score,
                        exec_status=execution_status,
                        has_submission=has_submission,
                        final_reward=reward,
                        prompt_preview=base_system_prompt[:100]
                    )
                
                print(f"[Adapter] Step {steps}: action_status='{status_str}', exec_status='{execution_status}', reward={reward:.4f}, is_success={is_success}")
                
                # Determine success based on execution status, not submission status
                # KaggleEnvironment returns obs with "action_status" and "execution_status" keys
                status_str = obs.get("action_status", "")
                execution_status = obs.get("execution_status", "")
                
                # Check if the code executed without errors (even if no submission was produced)
                # We consider it a success if:
                # 1. The overall status is SUCCESS (execution AND submission both successful), OR
                # 2. The reward is positive (got a submission score), OR
                # 3. The execution itself succeeded (no syntax/runtime errors), even if no submission yet
                execution_succeeded = (execution_status == "SUCCESS")
                
                is_success = (status_str == "SUCCESS" or reward > 0.0 or execution_succeeded)
                
                # Extract CV score from stdout if no submission reward
                # This is critical for GEPA optimization when agents explore solutions without submissions
                print(f"[Adapter Debug] Attempting CV extraction - execution_succeeded={execution_succeeded}, reward={reward}")
                print(f"[Adapter Debug] Obs type: {type(obs)}, Obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
                
                # Print feedback structure for debugging
                if isinstance(obs, dict) and "feedback" in obs:
                    feedback = obs["feedback"]
                    print(f"[Adapter Debug] Feedback type: {type(feedback)}")
                    if isinstance(feedback, dict):
                        print(f"[Adapter Debug] Feedback keys: {list(feedback.keys())[:5]}")  # Limit to first 5
                        # Check for execution key
                        if "execution" in feedback:
                            print(f"[Adapter Debug] Found 'execution' in feedback")
                            exec_data = feedback["execution"]
                            if isinstance(exec_data, dict):
                                print(f"[Adapter Debug] Execution keys: {list(exec_data.keys())}")
                    elif isinstance(feedback, str):
                        print(f"[Adapter Debug] Feedback is string: {feedback[:100]}")
                
                cv_score = self._extract_cv_score_from_output(obs, execution_succeeded)
                
                # Use CV score as reward if found and no submission reward exists
                if cv_score > 0 and reward == 0.0:
                    reward = cv_score
                    print(f"[Adapter] ✓ Using CV score from iteration output as reward: {reward:.4f}")
                
                print(f"[Adapter] Step {steps}: action_status='{status_str}', exec_status='{execution_status}', reward={reward:.4f}, is_success={is_success}")
                
                # Convert Gym output to the dictionary format Agent.parse_exec_result expects
                return {
                    "action_status": "SUCCESS" if is_success else "FAILED",
                    "feedback": obs,  # Pass full observation dict as feedback
                    "current_raw_score": info.get("raw_score", reward) if isinstance(info, dict) else reward,
                    "current_position_score": reward,  # Reward is the position score
                }

            # 3. The Agent Loop
        scores = [float(rollout["score"]) for rollout in rollout_outputs]
        trajectories = full_traces if capture_traces else None
        
        print(f"[Adapter] Returning {len(rollout_outputs)} outputs with scores: {scores}")
        print(f"[Adapter] CV scores logged to: {self.cv_log_file}")
        
        # Increment GEPA iteration counter for next evaluation
        self.gepa_iteration += 1
        self.total_evaluations += len(rollout_outputs)
        
        return EvaluationBatch(
            outputs=rollout_outputs,
            scores=scores,
            trajectories=trajectories
        )           print(f"[Adapter] Agent crashed: {str(e)}")
                    print(f"[Adapter] Traceback:\n{error_detail}")
                    episode_trace.append(f"Agent Crashed: {str(e)}\n{error_detail}")
                    break

            # 4. Finalize
            # We use the final reward from the last step as the score for this candidate
            final_score = 0.0
            
            print(f"[Adapter] Episode {episode_idx}: Agent generated {len(agent.journal.nodes)} nodes, Steps taken: {steps}")
            
            if agent.journal.nodes:
                # Debug: print all node scores
                print(f"[Adapter] All nodes in journal:")
                for i, node in enumerate(agent.journal.nodes):
                    node_score = node.metric.value if (hasattr(node, 'metric') and node.metric) else "No metric"
                    node_status = node.status if hasattr(node, 'status') else "No status"
                    is_buggy = node.is_buggy if hasattr(node, 'is_buggy') else "Unknown"
                    print(f"  Node {i}: score={node_score}, status={node_status}, buggy={is_buggy}")
                
                # Retrieve the best score recorded in the agent's journal
                best_node = agent.journal.get_best_node(only_good=False)  # Get best even if buggy
                if best_node:
                    if hasattr(best_node, 'metric') and best_node.metric:
                        final_score = float(best_node.metric.value)
                        print(f"[Adapter] Best node score: {final_score}")
                    else:
                        print(f"[Adapter] Best node has no metric, using 0.0")
                else:
                    print(f"[Adapter] No best node found, using 0.0")
            else:
                print(f"[Adapter] No nodes in journal, agent didn't execute successfully")

            total_score += final_score
            trace_str = "\n".join(episode_trace) if capture_traces else ""
            full_traces.append(trace_str)
            
            # Store rollout output for this episode
            rollout_outputs.append({
                "score": final_score,
                "trace": trace_str,
                "metadata": {"competition": self.competition_name, "episode": episode_idx}
            })
            
            env.close()

        # Return EvaluationBatch as expected by GEPA
        # EvaluationBatch expects: outputs (list of rollout outputs), scores (list of floats), trajectories (optional)
        scores = [float(rollout["score"]) for rollout in rollout_outputs]
        trajectories = full_traces if capture_traces else None
        
        print(f"[Adapter] Returning {len(rollout_outputs)} outputs with scores: {scores}")
        
        return EvaluationBatch(
            outputs=rollout_outputs,
            scores=scores,
            trajectories=trajectories
        )
    
    def _extract_cv_score_from_output(self, obs: dict, execution_succeeded: bool) -> float:
        """
        Extract cross-validation score from execution stdout.
        
        This method parses the agent's code execution output to find CV scores,
        which provides a reward signal even when no submission.csv is created.
        This is essential for GEPA optimization to evaluate intermediate solutions.
        
        Args:
            obs: Observation dictionary from environment step
            execution_succeeded: Whether code execution succeeded (no syntax/runtime errors)
            
        Returns:
            CV score if found, 0.0 otherwise
        """
        if not execution_succeeded:
            return 0.0
            
        import re
        
        # The observation now includes raw_result which contains the unprocessed execution output
        stdout_text = None
        
        # Method 1: Check raw_result (NEW - most direct path to stdout)
        if "raw_result" in obs and isinstance(obs["raw_result"], dict):
            raw_result = obs["raw_result"]
            if "execution" in raw_result and isinstance(raw_result["execution"], dict):
                exec_result = raw_result["execution"]
                # Check both 'output' (from utils.py) and 'stdout' (from sandbox.py)
                stdout_text = exec_result.get("output") or exec_result.get("stdout")
                if stdout_text:
                    print(f"[Adapter] Found stdout in raw_result->execution->output/stdout")
        
        # Method 2: Check if feedback is directly in obs (fallback for backwards compatibility)
        if not stdout_text and "feedback" in obs and isinstance(obs["feedback"], dict):
            feedback = obs["feedback"]
            
            # Check for nested feedback structure with modes
            for feedback_mode, feedback_data in feedback.items():
                if not isinstance(feedback_data, dict):
                    continue
                
                # Try raw_results path first (from feedback manager)
                if "raw_results" in feedback_data:
                    raw_results = feedback_data["raw_results"]
                    if isinstance(raw_results, dict) and "execution" in raw_results:
                        exec_result = raw_results["execution"]
                        if "stdout" in exec_result:
                            stdout_text = exec_result["stdout"]
                            break
                        elif "output" in exec_result:
                            stdout_text = exec_result["output"]
                            break
            
            # If not found in nested structure, check direct execution key
            if not stdout_text and "execution" in feedback:
                exec_result = feedback["execution"]
                if isinstance(exec_result, dict):
                    stdout_text = exec_result.get("stdout") or exec_result.get("output")
        
        # Method 3: Check if obs itself has execution key (alternative structure)
        if not stdout_text and "execution" in obs:
            exec_result = obs["execution"]
            if isinstance(exec_result, dict):
                stdout_text = exec_result.get("stdout") or exec_result.get("output")
        
        if not stdout_text:
            print(f"[Adapter] Could not find stdout in observation structure")
            return 0.0
        
        print(f"[Adapter] Found stdout, searching for CV score (length: {len(stdout_text)})")
        print(f"[Adapter] Stdout preview: {stdout_text[:200]}")
        
        # Parse CV score patterns from stdout
        # Common patterns:
        # - "Cross-validated accuracy: 0.8283"
        # - "5-fold cross-validation accuracy: 0.8137"
        # - "Cross-validation score: 0.8137"
        # - "CV accuracy: 0.8137"
        # - "Accuracy: 0.8137" (fallback)
        patterns = [
            r'cross[-\s]?validated?\s+(?:accuracy|score)[:\s]+([0-9.]+)',
            r'(?:cv|validation)\s+(?:accuracy|score)[:\s]+([0-9.]+)',
            r'(?:test|val)\s+(?:accuracy|score)[:\s]+([0-9.]+)',
            r'accuracy[:\s]+([0-9.]+)',
            r'score[:\s]+([0-9.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                cv_score = float(match.group(1))
                # Sanity check: scores should be between 0 and 1 (or 0-100 if percentage)
                if cv_score > 1.0:
                    cv_score = cv_score / 100.0  # Convert percentage to decimal
                if 0.0 <= cv_score <= 1.0:
                    print(f"[Adapter] ✓ Extracted CV score: {cv_score:.4f} using pattern: {pattern}")
                    return cv_score
        
        print(f"[Adapter] No CV score pattern matched in stdout")
        return 0.0