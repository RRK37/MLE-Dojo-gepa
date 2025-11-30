"""
Prompt templates and utilities for the MLE-STAR agent.
"""

from typing import Dict, Any, List, Optional
import yaml
import os

class MLEStarPrompts:
    """Prompt templates for the MLE-STAR agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        self.config = self._load_config(config_path)
        self._init_prompts()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _init_prompts(self) -> None:
        """Initialize all prompt templates."""
        # System prompt that sets the behavior of the agent
        self.system_prompt = """You are MLE-STAR, an advanced Machine Learning Engineering assistant specializing in Search and Targeted Refinement.

Your capabilities include:
1. Data analysis and preprocessing
2. Feature engineering and selection
3. Model architecture design
4. Hyperparameter optimization
5. Model evaluation and interpretation
6. Code generation and refinement

Guidelines:
- Be precise and methodical in your approach
- Explain your reasoning clearly
- Generate production-quality, efficient, and well-documented code
- Consider computational efficiency and resource constraints
- Always validate your approach with appropriate metrics
- Follow best practices in ML engineering and software development
- Be aware of potential data leakage and other common ML pitfalls
"""

        # Task analysis prompt
        self.task_analysis_prompt = """Analyze the following machine learning task and provide a structured plan:

Task: {task_description}

Available data:
{data_summary}

Please provide a structured plan with the following sections:
1. Problem understanding: Type of problem, key challenges, success criteria
2. Data preprocessing: Required cleaning, handling missing values, feature engineering ideas
3. Modeling approach: Potential models to try, architecture ideas, hyperparameters to tune
4. Evaluation: Metrics to track, validation strategy, potential pitfalls
5. Next steps: Immediate actions to take

Format your response in markdown with clear section headers.
"""

        # Code generation prompt
        self.code_generation_prompt = """Generate Python code for the following task:

Task: {task_description}

Requirements:
{requirements}

Additional context:
{context}

Please provide:
1. A clear explanation of your approach
2. Well-commented, production-quality Python code
3. Any necessary imports and setup
4. Example usage if applicable

Format your response as a markdown code block with language specification.
"""

        # Code refinement prompt
        self.refinement_prompt = """Refine the following code based on the feedback:

Original Code:
```python
{code}
```

Feedback:
{feedback}

Please provide:
1. An explanation of the changes made
2. The refined code with improvements
3. Any additional considerations or next steps

Format your response as a markdown code block with language specification.
"""

        # Error analysis prompt
        self.error_analysis_prompt = """Analyze the following error and suggest fixes:

Error:
{error_message}

Code context:
{code_context}

Please provide:
1. The root cause of the error
2. Step-by-step instructions to fix it
3. The corrected code
4. Suggestions to prevent similar issues
"""

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a formatted prompt by name with provided variables."""
        if not hasattr(self, prompt_name):
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        template = getattr(self, prompt_name)
        return template.format(**kwargs)

# Global instance for easy access
prompts = MLEStarPrompts()
