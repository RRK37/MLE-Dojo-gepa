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
            
            obs, _ = env.reset()
            done = False
            steps = 0
            
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
                obs, reward, terminated, truncated, info = env.step(action="execute_code", code=code_to_execute)
                
                done = terminated or truncated
                steps += 1
                
                # Log the result
                feedback_str = str(obs.get("feedback", obs))
                episode_trace.append(f"Execution Output:\n{feedback_str[:500]}...")
                
                # Determine success based on execution status in observation
                status_str = obs.get("status", "FAILED")
                is_success = status_str == "SUCCESS" or status_str == "success"
                
                # Convert Gym output to the dictionary format Agent.parse_exec_result expects
                return {
                    "action_status": "SUCCESS" if is_success else "FAILED",
                    "feedback": obs,  # Pass full observation dict as feedback
                    "current_raw_score": info.get("raw_score", reward),
                    "current_position_score": reward,  # Reward is the position score
                }

            # 3. The Agent Loop
            while not done and steps < self.max_steps:
                try:
                    # The agent plans and generates code, then calls our bridge
                    agent.step(exec_callback=env_bridge_callback)
                except Exception as e:
                    episode_trace.append(f"Agent Crashed: {str(e)}")
                    break

            # 4. Finalize
            # We use the final reward from the last step as the score for this candidate
            final_score = 0.0
            
            print(f"[Adapter] Episode {episode_idx}: Agent generated {len(agent.journal.nodes)} nodes, Steps taken: {steps}")
            
            if agent.journal.nodes:
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