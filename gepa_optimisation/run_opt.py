import sys
import os
import copy

# 1. Setup Path to find your agent code
# Assuming this file is in project_root/gepa_optimization/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 2. Imports
from gepa import optimize
from gepa import OptimizerConfig
from gepa_optimization.adapter import MLEDojoGEPAAdapter

# Import your agent and its dependencies
from existing_agent_code.agent import Agent
from mledojo.agent.aide.journal import Journal
from mledojo.agent.aide.utils.config import Config

# 3. Configuration & Factory
def main():
    # --- A. Setup Static Configuration ---
    # Load your config (mocking typical values here, adjust to your needs)
    # If you use hydra, load it properly here.
    cfg = Config(
        agent={"code": {"model_name": "gpt-4o", "model_mode": "gpt", "api_key": os.getenv("OPENAI_API_KEY")},
               "search": {"num_drafts": 2, "debug_prob": 0.2, "max_debug_depth": 3},
               "expose_prediction": True, "k_fold_validation": 5},
        exec={"timeout": 600},
        workspace_dir="./workspace"
    )
    
    task_desc = "Predict house prices based on the provided dataset. Optimize for RMSE."
    task_name = "mle-dojo/house-prices-v1" # Replace with your target env ID
    data_dir = "./input"
    output_dir = "./output"

    # --- B. Define the Factory ---
    # This ensures every GEPA trial gets a fresh Agent with a clean Journal
    def agent_factory(system_prompt: str):
        # 1. Create a fresh Journal for this episode
        fresh_journal = Journal() 
        
        # 2. Instantiate the agent with the injected prompt
        return Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=fresh_journal,
            higher_is_better=True, # Depends on metric (RMSE is False, Accuracy is True)
            data_dir=data_dir,
            output_dir=output_dir,
            system_prompt=system_prompt # <--- The GEPA optimization target
        )

    # --- C. Initialize Adapter ---
    adapter = MLEDojoGEPAAdapter(
        task_name=task_name,
        agent_factory=agent_factory,
        max_steps=8 # Limit steps to save tokens during optimization
    )

    # --- D. Configure GEPA ---
    gepa_config = OptimizerConfig(
        num_generations=5,          # How many rounds of evolution
        candidates_per_gen=4,       # How many variations per round
        reflection_model="gpt-4o",  # The LLM analyzing the traces
        evolution_strategy="pareto",
        temperature=0.7
    )

    initial_prompt = "You are a Kaggle Grandmaster. Focus on feature engineering and robust validation."

    # --- E. Run Optimization ---
    print(f"Starting GEPA Optimization on {task_name}...")
    
    result = optimize(
        adapter=adapter,
        initial_text_components={'system_prompt': initial_prompt},
        config=gepa_config
    )

    # --- F. Report Results ---
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Score: {result.best_score}")
    print("Best System Prompt:")
    print("-" * 20)
    print(result.best_candidate.text_components['system_prompt'])
    print("-" * 20)
    
    # Optional: Save the best prompt to a file
    with open("best_prompt.txt", "w") as f:
        f.write(result.best_candidate.text_components['system_prompt'])

if __name__ == "__main__":
    main()