"""
MLE-STAR Prompts - All prompts from PaperMethod.md (1-14)
Consolidated into a single file.
"""

from typing import Dict, List, Any, Optional


class MLEStarPrompts:
    """All MLE-STAR prompts from the paper methodology."""
    
    @staticmethod
    def prompt_1_model_retrieval(task_description: str, M: int = 3) -> str:
        """Prompt 1: Model retrieval from web search."""
        return f"""# Competition
{task_description}

# Your task
- List {M} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""

    @staticmethod
    def prompt_2_initial_solution(task_description: str, model_description: str, example_code: str, data_dir: str, available_packages: str = "", data_preview: str = "") -> str:
        """Prompt 2: Generate initial solution from model."""
        data_overview = f"\n# Data Overview:\n{data_preview}\n" if data_preview else ""
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
## Model name
{model_description}

## Example Python code
{example_code}
{data_overview}
# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- **CRITICAL: Data Loading Instructions:**
  - **Code runs from /tmp/ directory, NOT from the workspace**
  - **You MUST use the absolute path: `{data_dir}`**
  - **First, check what files exist: `import os; files = os.listdir('{data_dir}'); print('Available files:', files)`**
  - **Then load the training data file (e.g., `train.csv`, `train.parquet`, or similar)**
  - **Example: `train_data = pd.read_csv('{data_dir}/train.csv')` or `pd.read_parquet('{data_dir}/train.parquet')`**
  - **DO NOT use relative paths like `./input/train.csv` - they will fail!**
  - **Always verify the file exists before loading: `if os.path.exists('{data_dir}/train.csv'): ...`**
- Do not include other models that are not directly related to the model described.
- {available_packages}
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data from `{data_dir}`. Do not load test data.
- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

    @staticmethod
    def prompt_3_merge_solutions(task_description: str, base_code: str, reference_code: str, data_dir: str, available_packages: str = "") -> str:
        """Prompt 3: Merge base and reference solutions."""
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
{base_code}

# Reference solution
{reference_code}

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- **CRITICAL: Data Loading Instructions:**
  - **Code runs from /tmp/ directory, NOT from the workspace**
  - **You MUST use the absolute path: `{data_dir}`**
  - **First, check what files exist: `import os; files = os.listdir('{data_dir}'); print('Available files:', files)`**
  - **Then load the training data file (e.g., `train.csv`, `train.parquet`, or similar)**
  - **Example: `train_data = pd.read_csv('{data_dir}/train.csv')` or `pd.read_parquet('{data_dir}/train.parquet')`**
  - **DO NOT use relative paths like `./input/train.csv` - they will fail!**
  - **Always verify the file exists before loading: `if os.path.exists('{data_dir}/train.csv'): ...`**
- {available_packages}
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data from `{data_dir}`. Do not load test data.
- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

    @staticmethod
    def prompt_4_ablation_study(solution_script: str, previous_ablations: List[str], available_packages: str = "") -> str:
        """Prompt 4: Generate ablation study."""
        prev_ablation_text = ""
        for i, prev_ablation in enumerate(previous_ablations):
            prev_ablation_text += f"\n## Previous ablation study result {i}\n{prev_ablation}\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.
- We will also provide the summaries of previous ablation studies.

# Python solution
{solution_script}
{prev_ablation_text}

# Instructions
- You need you to generate a simple Python code that performs an ablation study on the train.py script.
- The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.
- Your ablation study should concentrate on the other parts that have not been previously considered.
- For each ablation, print out how the modification affects the model's performance.
- {available_packages}

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out what part of the code contributes the most to the overall performance."""

    @staticmethod
    def prompt_5_summarize_ablation(ablation_code: str, raw_result: str) -> str:
        """Prompt 5: Summarize ablation study results."""
        return f"""# Your code for ablation study was:
{ablation_code}

# Ablation study results after running the above code:
{raw_result}

# Your task
- Summarize the result of ablation study based on the code and printed output."""

    @staticmethod
    def prompt_6_extract_refine_plan(solution_script: str, ablation_summary: str, prev_code_blocks: List[str]) -> str:
        """Prompt 6: Extract code block and create refinement plan."""
        prev_blocks_text = ""
        for i, block in enumerate(prev_code_blocks):
            prev_blocks_text += f"\n## Code block {i}\n{block}\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.
- We also provide code blocks which you have tried to improve previously.

# Python solution
{solution_script}

# Ablation study results
{ablation_summary}
{prev_blocks_text}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Try to improve the other part which was not considered before.
- Also extract the code block from the above Python script that need to be improved according to the proposed plan. You should try to extract the code block which was not improved before.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:
Refine_Plan = {{'code_block': str, 'plan': str}}
Return: list[Refine_Plan]"""

    @staticmethod
    def prompt_7_refine_code_block(code_block: str, plan: str, available_packages: str = "") -> str:
        """Prompt 7: Refine code block based on plan."""
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need refine the code block for better performance based on the improvement plan.
- We will now provide the code block and the improvement plan.

# Code block
{code_block}

# Improvement plan
{plan}

# Your task
- Implement the improvement plan on the above code block. But do not remove subsampling if exists.
- The code block should be improved according to the proposed plan.
- Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables.
- {available_packages}

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the improved code block.
- There should be no additional headings or text in your response."""

    @staticmethod
    def prompt_8_alternative_plan(code_block: str, plans: List[str], scores: List[float]) -> str:
        """Prompt 8: Suggest alternative refinement plan."""
        plans_text = ""
        for i, (plan, score) in enumerate(zip(plans, scores)):
            plans_text += f"\n## Plan: {plan}\n## Score: {score}\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to improve the code block for better performance.
- We will provide the code block you are improving and the improvement plans you have tried.

# Code block
{code_block}
{plans_text}

# Your task
- Suggest a better plan to improve the above code block.
- The suggested plan must be novel and effective.
- Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- The suggested plan should be differ from the previous plans you have tried and should receive a higher score.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- There should be no additional headings or text in your response."""

    @staticmethod
    def prompt_9_ensemble_plan(solutions: List[str], plans: List[str], scores: List[float]) -> str:
        """Prompt 9: Suggest ensemble plan."""
        solutions_text = ""
        for i, solution in enumerate(solutions):
            solutions_text += f"\n# {i+1}st Python Solution\n{solution}\n"
        
        plans_text = ""
        for i, (plan, score) in enumerate(zip(plans, scores)):
            plans_text += f"\n## Plan: {plan}\n## Score: {score}\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {len(solutions)} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{solutions_text}
# Ensemble plans you have tried
{plans_text}

# Your task
- Suggest a better plan to ensemble the {len(solutions)} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be differ from the previous plans you have tried and should receive a higher (or lower) score.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur."""

    @staticmethod
    def prompt_10_implement_ensemble(solutions: List[str], plan: str, data_dir: str, available_packages: str = "") -> str:
        """Prompt 10: Implement ensemble."""
        solutions_text = ""
        for i, solution in enumerate(solutions):
            solutions_text += f"\n# {i+1}st Python Solution\n{solution}\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {len(solutions)} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{solutions_text}
# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much."
- **CRITICAL: Data Loading Instructions:**
  - **Code runs from /tmp/ directory, NOT from the workspace**
  - **You MUST use the absolute path: `{data_dir}`**
  - **First, check what files exist: `import os; files = os.listdir('{data_dir}'); print('Available files:', files)`**
  - **Then load the training data file (e.g., `train.csv`, `train.parquet`, or similar)**
  - **Example: `train_data = pd.read_csv('{data_dir}/train.csv')` or `pd.read_parquet('{data_dir}/train.parquet')`**
  - **DO NOT use relative paths like `./input/train.csv` - they will fail!**
  - All the provided data (except previous submissions; do not load submissions) is already prepared and available in `{data_dir}`.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {len(solutions)} Python Solutions.
- There should be no additional headings or text in your response.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {len(solutions)} provided solutions.
- Do not forget the `./final/submission.csv` file.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is."""

    @staticmethod
    def prompt_11_debug(code: str, bug: str, data_dir: str, task_desc: str = "", data_preview: str = "", available_packages: str = "", bug_history: Optional[List[Dict]] = None) -> str:
        """Prompt 11: Debug code with error - Enhanced with full context."""
        bug_history_text = ""
        if bug_history:
            bug_history_text = "\n# Previous Debug Attempts (to avoid repeating fixes):\n"
            for i, bug_entry in enumerate(bug_history[-3:]):  # Show last 3 attempts
                bug_history_text += f"\n## Attempt {bug_entry.get('attempt', i+1)}:\n"
                bug_history_text += f"Error: {bug_entry.get('error', '')[:200]}...\n"
                bug_history_text += f"Fix attempted: {bug_entry.get('fix_attempted', '')[:200]}...\n"
        
        task_context = f"\n# Task Description:\n{task_desc}\n" if task_desc else ""
        data_context = f"\n# Data Overview:\n{data_preview}\n" if data_preview else ""
        
        return f"""# Code with an error:
{code}

# Error:
{bug}
{task_context}{data_context}{bug_history_text}
# Your task
- Analyze the root cause: Why did this error occur? What is the underlying issue?
- Please revise the code to fix the error.
- Do not repeat fixes that were already attempted (see previous debug attempts above).
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- **CRITICAL: Data Loading Instructions:**
  - **Code runs from /tmp/ directory, NOT from the workspace**
  - **You MUST use the absolute path: `{data_dir}`**
  - **First, check what files exist: `import os; files = os.listdir('{data_dir}'); print('Available files:', files)`**
  - **Then load the training data file (e.g., `train.csv`, `train.parquet`, or similar)**
  - **Example: `train_data = pd.read_csv('{data_dir}/train.csv')` or `pd.read_parquet('{data_dir}/train.parquet')`**
  - **DO NOT use relative paths like `./input/train.csv` - they will fail!**
  - **Always verify the file exists before loading: `if os.path.exists('{data_dir}/train.csv'): ...`**
- {available_packages}
- There should be no additional headings or text in your response.
- All the provided input data is stored in `{data_dir}` directory.
- **CRITICAL: Data Loading Instructions:**
  - **Code runs from /tmp/ directory, NOT from the workspace**
  - **You MUST use the absolute path: `{data_dir}`**
  - **First, check what files exist: `import os; files = os.listdir('{data_dir}'); print('Available files:', files)`**
  - **Then load the training data file (e.g., `train.csv`, `train.parquet`, or similar)**
  - **Example: `train_data = pd.read_csv('{data_dir}/train.csv')` or `pd.read_parquet('{data_dir}/train.parquet')`**
  - **DO NOT use relative paths like `./input/train.csv` - they will fail!**
  - **Always verify the file exists before loading: `if os.path.exists('{data_dir}/train.csv'): ...`**
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""

    @staticmethod
    def prompt_12_check_leakage(code: str) -> str:
        """Prompt 12: Check for data leakage."""
        return f"""# Python code
{code}

# Your task
- Extract the code block where the validation and test samples are preprocessed using training samples.
- Check that the model is trained with only training samples.
- Check that before printing the final validation score, the model is not trained the validation samples.
- Also check whether the validation and test samples are preprocessed correctly, preventing information from the validation or test samples from influencing the training process (i.e., preventing data leakage).

# Requirement
- Extract a code block and also check the data leakage.
- The code block should be an exact subset of the above Python code.
- Your response for a code block should be a single markdown code block.
- If data leakage is present on validation and test samples, answer 'Yes Data Leakage'.
- If data leakage is not present on validation and test samples, answer 'No Data Leakage'.

Use this JSON schema:
Answer = {{'leakage_status': str, 'code_block': str}}
Return: list[Answer]"""

    @staticmethod
    def prompt_13_fix_leakage(code: str) -> str:
        """Prompt 13: Fix data leakage."""
        return f"""# Python code
{code}

# Your task
- In the above Python code, the validation and test samples are influencing the training process, i.e., not correctly preprocessed.
- Ensure that the model is trained with only training samples.
- Ensure that before printing the final validation score, the model is not trained on the validation samples.
- Refine the code to prevent such data leakage problem.

# Requirement
- Your response should be a single markdown code block.
- Note that all the variables are defined earlier. Just modify it with the above code."""

    @staticmethod
    def prompt_14_check_data_usage(initial_solution: str, task_description: str) -> str:
        """Prompt 14: Check if all data is used."""
        return f"""I have provided Python code for a machine learning task (attached below):

# Solution Code
{initial_solution}

Does above solution code uses all the information provided for training? Here is task description and some guide to handle:

# Task description
{task_description}

# Your task
- If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
- DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
- See the task description carefully, to know how to extract unused information effectively.
- When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

# Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response  
Option 2: If the code used all the provided information, simply state that 'All the provided information is used.'"""


# Global instance
prompts = MLEStarPrompts()
