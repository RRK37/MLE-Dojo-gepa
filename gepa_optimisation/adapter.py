import gymnasium as gym
from gepa.core import GEPAAdapter, Candidate
from typing import Callable, Any

class MLEDojoGEPAAdapter(GEPAAdapter):
    def __init__(self, task_name: str, agent_factory: Callable[[str], Any], max_steps: int = 10):
        """
        Args:
            task_name: The MLE-Dojo gym environment ID.
            agent_factory: A function that takes `system_prompt` and returns a FRESH Agent instance.
            max_steps: Safety limit for agent iterations per episode.
        """
        self.task_name = task_name
        self.agent_factory = agent_factory
        self.max_steps = max_steps

    def evaluate(self, candidate: Candidate, num_episodes: int = 1) -> dict:
        """
        Runs the agent against the environment using the candidate prompt.
        """
        # 1. Get the new prompt from GEPA
        system_prompt = candidate.text_components['system_prompt']
        
        # 2. Run Episodes
        total_score = 0
        full_traces = []

        for _ in range(num_episodes):
            # Create a fresh environment and agent for this run
            env = gym.make(self.task_name)
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
                
                # Execute in Gym
                # Note: MLE-Dojo envs typically expect code as the action
                obs, reward, terminated, truncated, info = env.step(code_to_execute)
                
                done = terminated or truncated
                steps += 1
                
                # Log the result
                feedback_str = str(obs)
                episode_trace.append(f"Execution Output:\n{feedback_str[:500]}...")
                
                # Convert Gym output to the dictionary format your Agent.parse_exec_result expects
                # Adjust keys 'action_status', 'feedback', 'current_position_score' based on your specific Agent logic
                return {
                    "action_status": "SUCCESS" if not "Error" in feedback_str else "FAILED",
                    "feedback": feedback_str,
                    "current_raw_score": info.get("score", 0.0),
                    "current_position_score": reward, # Assuming reward is the metric we care about
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
            # (Or you could query env.get_best_score() if supported)
            final_score = 0
            if agent.journal.nodes:
                # Retrieve the best score recorded in the agent's journal
                best_node = agent.journal.get_best_node()
                if best_node and best_node.metric:
                    final_score = best_node.metric.value

            total_score += final_score
            full_traces.append("\n".join(episode_trace))
            
            env.close()

        avg_score = total_score / num_episodes

        return {
            "score": avg_score,
            "traces": full_traces,
            "metadata": {"task": self.task_name}
        }