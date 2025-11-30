"""
Search-related prompts for MLE-STAR agent.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SearchPrompts:
    """Prompts for web search and research tasks."""
    
    @staticmethod
    def get_initial_search_prompt(task_description: str, competition_info: Dict[str, Any]) -> str:
        """Generate prompt for initial web search."""
        return f"""
        You are an expert ML researcher. Your task is to find the most relevant information for:
        
        Competition: {competition_info.get('title', 'N/A')}
        Task: {task_description}
        
        Search for:
        1. State-of-the-art models and approaches for this specific task
        2. Recent research papers with code implementations
        3. Winning solutions from similar competitions
        4. Common pitfalls and how to avoid them
        5. Recommended evaluation metrics and validation strategies
        
        Focus on practical, implementable solutions with available code.
        """

    @staticmethod
    def get_model_retrieval_prompt(task_description: str, search_results: List[Dict]) -> str:
        """Generate prompt for evaluating and selecting models from search results."""
        results_str = "\n".join([f"- {res['title']} - {res.get('snippet', '')}" for res in search_results])
        
        return f"""
        Evaluate the following search results for the task: {task_description}
        
        Search Results:
        {results_str}
        
        For each result, provide:
        1. Relevance score (1-10)
        2. Key insights
        3. Potential implementation approach
        
        Select the top 3 most promising approaches for implementation.
        """

    @staticmethod
    def get_ablation_study_prompt(current_solution: str, component_to_ablate: str) -> str:
        """Generate prompt for creating an ablation study."""
        return f"""
        Design an ablation study for the following component:
        Component: {component_to_ablate}
        
        Current solution:
        ```python
        {current_solution}
        ```
        
        Generate a Python script that:
        1. Creates a modified version of the solution with the component ablated/removed
        2. Includes appropriate evaluation metrics
        3. Compares performance with the original solution
        4. Provides clear, interpretable results
        
        Focus on scientific rigor and reproducibility.
        """

    @staticmethod
    def get_ablation_summary_prompt(ablation_results: Dict[str, Any]) -> str:
        """Generate a summary of ablation study results."""
        return f"""
        Analyze these ablation study results and provide a concise summary:
        
        {ablation_results}
        
        Include:
        1. Key findings
        2. Impact on model performance
        3. Recommendations for improvement
        4. Potential next steps
        """

    @classmethod
    def get_ensemble_prompt(cls, models: List[Dict]) -> str:
        """Generate prompt for creating an ensemble of models."""
        models_str = "\n".join([
            f"- {m['name']}: {m.get('description', '')} (Performance: {m.get('performance', 'N/A')})"
            for m in models
        ])
        
        return f"""
        Create an ensemble strategy for the following models:
        
        {models_str}
        
        Provide:
        1. Recommended ensemble method with justification
        2. Implementation code
        3. Expected benefits
        4. Potential challenges and mitigations
        """
