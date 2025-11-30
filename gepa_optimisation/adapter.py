from gepa import GEPAAdapter, EvaluationBatch
from typing import Callable, Any
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric

class MLEDojoGEPAAdapter(GEPAAdapter):
    def __init__(self, 
                 competition_name: str,
                 data_dir: str,
                 output_dir: str,
                 agent_factory: Callable[[str], Any], 
                 max_steps: int = 10,
                 execution_timeout: int = 600,
                 score_mode: str = "position"):
        """
        Args:
            competition_name: The Kaggle competition name (e.g., 'titanic').
            data_dir: Path to competition data directory.
            output_dir: Path to output directory for env results.
            agent_factory: A function that takes `system_prompt` and returns a FRESH Agent instance.
            max_steps: Safety limit for agent iterations per episode.
            execution_timeout: Timeout for code execution in seconds.
            score_mode: Scoring mode ('position' or 'raw').
        """
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.agent_factory = agent_factory
        self.max_steps = max_steps
        self.execution_timeout = execution_timeout
        self.score_mode = score_mode

    def evaluate(self, batch: list, candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """
        Runs the agent against the environment using the candidate prompt.
        """
        # 1. Get the new prompt from GEPA (candidate is a dict, not an object)
        system_prompt = candidate['system_prompt']
        
        print(f"\n[Adapter] Evaluating with system prompt: {system_prompt[:100]}...")
        print(f"[Adapter] Batch size: {len(batch)}, Capture traces: {capture_traces}")
        
        # 2. Run Episodes (1 episode per batch item, or default to 1 if batch is empty)
        num_episodes = len(batch) if batch else 1
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