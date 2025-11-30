"""
Utility functions for prompt management in GEPA integration.

This module provides helpers for extracting, injecting, validating, and serializing
prompt configurations.
"""

import json
from typing import Dict
from pathlib import Path


def extract_default_prompts() -> Dict[str, str]:
    """
    Extract the default prompts from AIDE agent.
    
    Returns:
        Dict mapping component names to default prompt text
    """
    return {
        'introduction_draft': (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        ),
        'introduction_improve': (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance, i.e. the position score provided by the competition. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        ),
        'introduction_debug': (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        ),
    }


def inject_prompts(candidate: Dict[str, str]) -> Dict[str, str]:
    """
    Inject and validate prompts from a GEPA candidate.
    
    Args:
        candidate: Dict from GEPA with prompt component text
        
    Returns:
        Validated prompt dict ready for AIDE agent
    """
    defaults = extract_default_prompts()
    
    # Start with defaults and override with candidate values
    prompts = defaults.copy()
    prompts.update(candidate)
    
    return prompts


def validate_prompts(prompts: Dict[str, str]) -> bool:
    """
    Validate that prompt configuration is complete and well-formed.
    
    Args:
        prompts: Dict of prompts to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['introduction_draft', 'introduction_improve', 'introduction_debug']
    
    for key in required_keys:
        if key not in prompts:
            raise ValueError(f"Missing required prompt component: {key}")
        
        if not isinstance(prompts[key], str):
            raise ValueError(f"Prompt component {key} must be a string")
        
        if len(prompts[key].strip()) == 0:
            raise ValueError(f"Prompt component {key} cannot be empty")
    
    return True


def serialize_prompts(prompts: Dict[str, str], filepath: str | Path) -> None:
    """
    Save prompts to a JSON file.
    
    Args:
        prompts: Dict of prompts to save
        filepath: Path where to save the prompts
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def deserialize_prompts(filepath: str | Path) -> Dict[str, str]:
    """
    Load prompts from a JSON file.
    
    Args:
        filepath: Path to the prompts JSON file
        
    Returns:
        Dict of prompts
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    validate_prompts(prompts)
    return prompts
