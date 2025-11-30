"""
Agent package for MLE-Dojo.

This package contains all the available agents that can be used with the MLE-Dojo framework.
"""

from typing import Dict, Type, Any

# Import all agents
from mledojo.agent.mleagent.agent import KaggleAgent as MLEAgent
from mledojo.agent.openaiagent.agent import OpenAIAgent
from mledojo.agent.aide.agent import Agent as AIDAgent
from mledojo.agent.dummy.agent import DummyAgent
from mledojo.agent.mlestar.agent import MLEStarAgent

# Agent registry
AGENT_REGISTRY: Dict[str, Type[Any]] = {
    'mle': MLEAgent,
    'openai': OpenAIAgent,
    'aide': AIDAgent,
    'dummy': DummyAgent,
    'mlestar': MLEStarAgent,  # Register MLE-STAR agent
}

def get_agent_class(agent_type: str):
    """Get agent class by type.
    
    Args:
        agent_type: Type of agent ('mle', 'openai', 'aide', 'dummy', 'mlestar')
        
    Returns:
        The agent class
        
    Raises:
        ValueError: If agent type is not found
    """
    agent_class = AGENT_REGISTRY.get(agent_type.lower())
    if agent_class is None:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available agents: {', '.join(AGENT_REGISTRY.keys())}"
        )
    return agent_class