from gepa import GEPAAdapter, EvaluationBatch
from typing import Callable, Any
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

class MLEDojoGEPAAdapter(GEPAAdapter):
    def __init__(self, 
                 competition_name: str,
                 data_dir: str,
                 output_dir: str,
                 agent_factory: Callable[[str], Any], 
                 max_steps: int = 10,
                 execution_timeout: int = 600,
                 score_mode: str = "position",
                 enable_live_plot: bool = True):
        """
        Args:
            competition_name: The Kaggle competition name (e.g., 'titanic').
            data_dir: Path to competition data directory.
            output_dir: Path to output directory for env results.
            agent_factory: A function that takes `system_prompt` and returns a FRESH Agent instance.
            max_steps: Safety limit for agent iterations per episode.
            execution_timeout: Timeout for code execution in seconds.
            score_mode: Scoring mode ('position' or 'raw').
            enable_live_plot: Whether to enable live plotting of journal nodes.
        """
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.agent_factory = agent_factory
        self.max_steps = max_steps
        self.execution_timeout = execution_timeout
        self.score_mode = score_mode
        self.enable_live_plot = enable_live_plot
        
        # Live plotting state
        self.fig = None
        self.ax = None
        self.plot_data = {'node_ids': [], 'scores': [], 'buggy': [], 'status': []}
        
        if self.enable_live_plot:
            self._setup_live_plot()

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
                
                done = terminated or truncated
                steps += 1
                
                # Log the result
                feedback_str = str(obs.get("feedback", obs))
                episode_trace.append(f"Execution Output:\n{feedback_str[:500]}...")
                
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
            while not done and steps < self.max_steps:
                try:
                    print(f"[Adapter] Calling agent.step() - iteration {steps + 1}/{self.max_steps}")
                    # The agent plans and generates code, then calls our bridge
                    agent.step(exec_callback=env_bridge_callback)
                    print(f"[Adapter] agent.step() completed, journal now has {len(agent.journal.nodes)} nodes")
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"[Adapter] Agent crashed: {str(e)}")
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
                
                # Update live plot with all nodes (in case we missed any)
                if self.enable_live_plot:
                    self._update_live_plot(agent.journal)
                
                # Save journal data to files
                self._save_journal_data(agent.journal, episode_idx)
                
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
    
    def _setup_live_plot(self):
        """Initialize the live plotting window with dark theme and pink/green styling."""
        plt.ion()  # Enable interactive mode
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('Journal Node Scores - Live')
        
        self.ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        self.ax.set_title('Journal Node Scores (Live)', fontsize=14, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def _update_live_plot(self, journal):
        """Update the live plot with current journal data."""
        if not self.enable_live_plot or self.ax is None:
            return
        
        # Clear the axis
        self.ax.clear()
        
        # Extract all node data
        node_ids = []
        scores = []
        buggy_flags = []
        
        for i, node in enumerate(journal.nodes):
            node_score = float(node.metric.value) if (hasattr(node, 'metric') and node.metric) else 0.0
            is_buggy = bool(node.is_buggy) if hasattr(node, 'is_buggy') else True
            
            node_ids.append(i)
            scores.append(node_score)
            buggy_flags.append(is_buggy)
        
        if not node_ids:
            return
        
        # Plot with pink (buggy) and green (successful) dots
        buggy_nodes = [(nid, score) for nid, score, buggy in zip(node_ids, scores, buggy_flags) if buggy]
        good_nodes = [(nid, score) for nid, score, buggy in zip(node_ids, scores, buggy_flags) if not buggy]
        
        if buggy_nodes:
            buggy_ids, buggy_scores = zip(*buggy_nodes)
            self.ax.scatter(buggy_ids, buggy_scores, 
                          c='#FF69B4', s=100, alpha=0.7, label='Buggy/Failed',
                          edgecolors='white', linewidth=0.5)
        
        if good_nodes:
            good_ids, good_scores = zip(*good_nodes)
            self.ax.scatter(good_ids, good_scores, 
                          c='#00FF7F', s=100, alpha=0.7, label='Successful',
                          edgecolors='white', linewidth=0.5)
        
        # Restore styling
        self.ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        self.ax.set_title(f'Journal Node Scores (Live) - {len(node_ids)} nodes', 
                         fontsize=14, fontweight='bold', pad=20)
        self.ax.legend(loc='best', framealpha=0.9)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _save_journal_data(self, journal, episode_idx: int):
        """
        Save journal node data to JSON and CSV files for later analysis and plotting.
        
        Args:
            journal: The agent's journal object containing node history
            episode_idx: Current episode index for file naming
        """
        # Create logs directory if it doesn't exist
        logs_dir = Path(self.output_dir) / "journal_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data structures
        json_data = {
            "competition": self.competition_name,
            "episode": episode_idx,
            "timestamp": timestamp,
            "nodes": []
        }
        
        csv_rows = []
        
        for i, node in enumerate(journal.nodes):
            # Extract node information
            node_score = float(node.metric.value) if (hasattr(node, 'metric') and node.metric) else 0.0
            node_status = str(node.status) if hasattr(node, 'status') else "UNKNOWN"
            is_buggy = bool(node.is_buggy) if hasattr(node, 'is_buggy') else False
            
            # JSON format (detailed)
            node_data = {
                "node_id": i,
                "score": node_score,
                "status": node_status,
                "buggy": is_buggy,
                "episode": episode_idx
            }
            json_data["nodes"].append(node_data)
            
            # CSV format (flat, easy for plotting)
            csv_rows.append({
                "episode": episode_idx,
                "node_id": i,
                "score": node_score,
                "status": node_status,
                "buggy": is_buggy,
                "timestamp": timestamp
            })
        
        # Save JSON file
        json_file = logs_dir / f"journal_episode_{episode_idx}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"[Adapter] Saved journal to {json_file}")
        
        # Save/append to CSV file (append mode for continuous logging)
        csv_file = logs_dir / f"journal_history.csv"
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='') as f:
            if csv_rows:
                fieldnames = ['episode', 'node_id', 'score', 'status', 'buggy', 'timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(csv_rows)
        
        print(f"[Adapter] Appended {len(csv_rows)} rows to {csv_file}")