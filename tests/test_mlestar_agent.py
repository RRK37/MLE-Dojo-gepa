"""
Test script for the MLE-STAR agent.

This script tests the basic functionality of the MLE-STAR agent
and its integration with the MLE-Dojo framework.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mledojo.agent.mleagent.agent import LLMConfig
from mledojo.agent.mlestar.agent import MLEStarAgent
from mledojo.gym.env import KaggleEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mlestar_agent():
    """Test the MLE-STAR agent with a simple Kaggle environment."""
    try:
        # Create a test configuration
        config = {
            'agent': {
                'model_mode': 'openai',  # or 'gemini', 'claude', etc.
                'model_name': 'gpt-4',   # or 'gemini-pro', 'claude-3-opus', etc.
                'api_key': os.getenv('OPENAI_API_KEY'),  # or GEMINI_API_KEY, etc.
                'max_completion_tokens': 2048,
                'max_prompt_tokens': 8192,
                'temperature': 0.7,
                'top_p': 0.9,
                'api_idx': 0,
                'history_length': 10,
                'init_method': 'cold'
            },
            'kaggle': {
                'work_dir': 'test_workspace',
                'competition': 'titanic'  # Simple competition for testing
            },
            'env': {
                'max_steps': 5,  # Limit to 5 steps for testing
                'execution_timeout': 300  # 5 minute timeout
            }
        }
        
        # Create LLM config
        llm_config = LLMConfig(
            model_mode=config['agent']['model_mode'],
            model_name=config['agent']['model_name'],
            max_completion_tokens=config['agent']['max_completion_tokens'],
            max_prompt_tokens=config['agent']['max_prompt_tokens'],
            temperature=config['agent']['temperature'],
            top_p=config['agent']['top_p']
        )
        
        # Initialize the agent
        logger.info("Initializing MLE-STAR agent...")
        agent = MLEStarAgent(
            api_idx=config['agent']['api_idx'],
            api_key=config['agent']['api_key'],
            llm_config=llm_config,
            history_length=config['agent']['history_length'],
            init_method=config['agent']['init_method'],
            config=config
        )
        
        # Create a test environment
        logger.info("Creating test environment...")
        env = KaggleEnvironment(
            competition=config['kaggle']['competition'],
            data_dir=os.path.join('data', config['kaggle']['competition']),
            output_dir=config['kaggle'].get('work_dir', 'test_workspace'),
            max_steps=config['env']['max_steps'],
            execution_timeout=config['env']['execution_timeout']
        )
        
        # Run the agent
        logger.info("Starting agent execution...")
        obs = env.reset()
        done = False
        
        while not done:
            # Get action from agent
            action, params = agent.act(obs, env.steps_left, env.time_left)
            logger.info(f"Action: {action}, Params: {json.dumps(params, indent=2)}")
            
            # Execute action in environment
            obs, reward, done, info = env.step(action, **params)
            logger.info(f"Reward: {reward}, Done: {done}")
            
            # Print environment state
            logger.info(f"Steps left: {env.steps_left}, Time left: {env.time_left}s")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_mlestar_agent()
