import sys
import os
import copy

# 1. Setup Path to find your agent code
# Assuming this file is in project_root/gepa_optimization/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 2. Imports
from gepa import optimize
from gepa_optimisation.adapter import MLEDojoGEPAAdapter
from gepa_optimisation.extract_insights import GEPAInsightsExtractor
from omegaconf import OmegaConf

# Import your agent and its dependencies
from mledojo.agent.aide.agent import Agent
from mledojo.agent.aide.journal import Journal
from mledojo.agent.aide.utils.config import Config, StageConfig, SearchConfig, AgentConfig, ExecConfig

# 3. Configuration & Factory
def main():
    # --- A. Setup Static Configuration ---
    # Build config using OmegaConf for proper structure
    # Create necessary directories
    workspace_dir = os.path.abspath("./workspace")
    log_dir = os.path.abspath("./logs")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    cfg_dict = {
        "data_dir": "./input",
        "desc_file": None,
        "name": "gepa-optimization",
        "goal": "Optimize system prompt for best competition performance",
        "eval": None,
        "log_dir": log_dir,
        "workspace_dir": workspace_dir,
        "preprocess_data": False,
        "copy_data": False,
        "exp_name": "gepa_opt",
        "generate_report": False,
        "agent": {
            "steps": 10,
            "k_fold_validation": 5,
            "expose_prediction": True,
            "data_preview": True,
            "code": {
                "model_name": "gpt-4o",
                "model_mode": "gpt",
                "port": 8314,
                "max_completion_tokens": 8192,
                "max_prompt_tokens": 30000,
                "api_idx": 0,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.7,
                "top_p": None
            },
            "search": {
                "num_drafts": 2,
                "debug_prob": 0.2,
                "max_debug_depth": 3
            }
        },
        "exec": {
            "timeout": 600,
            "agent_file_name": None,
            "format_tb_ipython": False
        }
    }
    cfg_schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, OmegaConf.create(cfg_dict))
    
    # Competition and environment settings
    competition_name = "titanic"  # Replace with your competition name
    # Note: data_dir must point to the directory containing public/ and private/ subdirectories
    # In the prepared data structure, this is at data/prepared/{competition}/data/
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "prepared", competition_name, "data")
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath("./output")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    task_desc = "Predict survival on the Titanic. Optimize for accuracy."
    
    # Workaround: Fix Titanic data structure to match framework expectations
    import shutil
    import pandas as pd
    from pathlib import Path
    
    # 1. Create sample_submission.csv from gender_submission.csv
    sample_sub_path = Path(data_dir) / "public" / "sample_submission.csv"
    gender_sub_path = Path(data_dir) / "public" / "gender_submission.csv"
    if not sample_sub_path.exists() and gender_sub_path.exists():
        print(f"Creating sample_submission.csv from gender_submission.csv...")
        shutil.copy(gender_sub_path, sample_sub_path)
    
    # 2. Create test_answer.csv - read test.csv and add ground truth labels
    test_answer_path = Path(data_dir) / "private" / "test_answer.csv"
    if not test_answer_path.exists():
        print(f"Creating test_answer.csv with ground truth labels...")
        # Read test.csv to get PassengerIds
        test_csv_path = Path(data_dir) / "public" / "test.csv"
        if test_csv_path.exists():
            test_df = pd.read_csv(test_csv_path)
            # For Titanic, we need to create ground truth labels
            # Since we don't have actual labels, we'll use the gender_submission.csv as a baseline
            gender_sub_path = Path(data_dir) / "public" / "gender_submission.csv"
            if gender_sub_path.exists():
                answer_df = pd.read_csv(gender_sub_path)
                answer_df.to_csv(test_answer_path, index=False)
                print(f"Created test_answer.csv with {len(answer_df)} rows")
            else:
                # Fallback: create dummy predictions based on passenger ID parity
                answer_df = pd.DataFrame({
                    'PassengerId': test_df['PassengerId'],
                    'Survived': (test_df['PassengerId'] % 2).astype(int)  # Dummy labels
                })
                answer_df.to_csv(test_answer_path, index=False)
                print(f"Created test_answer.csv with dummy labels ({len(answer_df)} rows)")
        else:
            print(f"Warning: Could not create test_answer.csv - test.csv not found")

    # --- B. Define the Factory ---
    # This ensures every GEPA trial gets a fresh Agent with a clean Journal
    def agent_factory(system_prompt: str):
        # 1. Create a fresh Journal for this episode
        fresh_journal = Journal() 
        
        # 2. Agent needs data_dir pointing to public data subdirectory
        agent_data_dir = os.path.join(data_dir, "public")
        
        # 3. Instantiate the agent with the injected prompt
        return Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=fresh_journal,
            higher_is_better=True, # Depends on metric (RMSE is False, Accuracy is True)
            data_dir=agent_data_dir,  # Point to public/ subdirectory where train.csv, test.csv are
            output_dir=output_dir,
            system_prompt=system_prompt # <--- The GEPA optimization target
        )

    # --- C. Setup Incremental Insights Saving ---
    insights_dir = os.path.abspath("./gepa_insights")
    os.makedirs(insights_dir, exist_ok=True)
    extractor = GEPAInsightsExtractor(output_dir=insights_dir)
    
    def save_incremental_insights(history, candidate, score):
        """Callback to save insights after each evaluation."""
        # Create a minimal result object for extraction
        class IncrementalResult:
            def __init__(self, history_list, current_candidate, current_score):
                self.best_candidate = current_candidate
                self.best_score = current_score
                self.history = history_list
        
        result = IncrementalResult(history, candidate, score)
        print(f"\n[Incremental Save] Saving insights after evaluation (score: {score:.4f})...")
        try:
            extractor.extract_and_save(result, competition_name=competition_name)
            print(f"[Incremental Save] ✓ Insights saved to {insights_dir}")
        except Exception as e:
            print(f"[Incremental Save] ✗ Failed to save insights: {e}")
    
    # --- D. Initialize Adapter ---
    adapter = MLEDojoGEPAAdapter(
        competition_name=competition_name,
        data_dir=data_dir,
        output_dir=output_dir,
        agent_factory=agent_factory,
        max_steps=3,  # Minimal steps for quick testing
        execution_timeout=600,
        score_mode="position",
        insights_callback=save_incremental_insights
    )

    # --- E. Configure Initial Prompt ---
    # Simpler initial prompt - easier for GEPA to mutate and improve
    # GEPA will add strategies, techniques, and specific instructions through mutations
    initial_prompt = """You are a Kaggle expert. Your goal is to maximize the competition score.

Key priorities:
- Build a strong baseline model first
- Use cross-validation to evaluate improvements
- Create informative features from the available data
- Generate a valid submission.csv file

Focus on writing clean, executable code that improves the score."""
    
    seed_candidate = {'system_prompt': initial_prompt}

    # --- F. Run Optimization ---
    print(f"Starting GEPA Optimization on {competition_name}...")
    
    # GEPA requires a non-empty trainset for its reflection/sampling mechanisms
    # We provide dummy items since the adapter handles episodes internally
    trainset = [{"episode": i} for i in range(1)]  # 1 episode per evaluation (reduced for quick testing)
    
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,  # Provide dummy trainset items for GEPA's batch sampler
        adapter=adapter,
        reflection_lm="gpt-4o",  # The LLM analyzing the traces
        candidate_selection_strategy="current_best",  # Select best performing candidate
        max_metric_calls=6,  # Increased to 6 for more mutation opportunities (1 seed + 5 mutations)
        display_progress_bar=True,
        temperature=1.0  # Higher temperature for more creative/aggressive mutations
    )

    # --- G. Report Results ---
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Score: {result.best_score}")
    print("Best System Prompt:")
    print("-" * 20)
    print(result.best_candidate['system_prompt'])
    print("-" * 20)
    
    # --- H. Extract and Save Final Insights ---
    print("\n" + "="*50)
    print("EXTRACTING FINAL OPTIMIZATION INSIGHTS...")
    print("="*50)
    
    insights = extractor.extract_and_save(result, competition_name=competition_name)
    
    print("\n" + "="*50)
    print(f"✓ Insights saved to: {insights_dir}")
    print("  - JSON format: gepa_insights_*.json")
    print("  - Human-readable: gepa_report_*.txt")
    print("="*50)
    
    # Optional: Save the best prompt to a quick-access file
    with open("best_prompt.txt", "w") as f:
        f.write(result.best_candidate['system_prompt'])
    print(f"✓ Best prompt saved to: best_prompt.txt")

if __name__ == "__main__":
    main()