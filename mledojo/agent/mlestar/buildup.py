"""
Setup module for MLE-STAR Agent.

This module provides functions to set up and configure the MLE-STAR agent
for Kaggle competitions. It handles loading configurations, preparing
workspaces, and initializing the agent with appropriate parameters.

The module is designed to work with the main.py script and integrates
with the MLE-Dojo framework for running AI agents on Kaggle competitions.
"""

import os
import logging
from typing import Dict, Any, Tuple

from mledojo.agent.mlestar.agent import MLESTARAgent, MLEStarConfig
from mledojo.agent.aide.journal import Journal
from mledojo.agent.aide.utils.config import (
    _load_cfg,
    prep_cfg,
    load_task_desc,
    prep_agent_workspace,
)
from mledojo.utils import get_metric

from rich.status import Status

logger = logging.getLogger("mlestar_setup")

def setup_mlestar_agent(
    config: Dict[str, Any], 
) -> Tuple[MLESTARAgent, Journal, Any]:
    """
    Set up an MLE-STAR Agent based on configuration.
    
    This function initializes and configures an MLE-STAR agent for a specific
    Kaggle competition. It loads the configuration, prepares the workspace,
    and initializes the agent with the appropriate parameters.
    
    Args:
        config: Configuration dictionary containing agent and competition settings
        
    Returns:
        Tuple containing:
            - Configured MLE-STAR Agent instance
            - Journal for tracking agent progress
            - Configuration object for the agent
    """
    # Load and prepare AIDE config (MLE-STAR extends AIDE)
    _cfg = _load_cfg(use_cli_args=False)
    data_dir = config['competition']['data_dir']
    desc_file = os.path.join(data_dir, "public", "description.txt")
    
    # Configure paths and settings
    _cfg.data_dir = os.path.join(data_dir, "public")
    _cfg.name = config['competition']['name']
    _cfg.desc_file = desc_file
    _cfg.log_dir = os.path.join(config['output_dir'], "logs")
    _cfg.workspace_dir = config['output_dir']
    
    # Prepare the configuration
    cfg = prep_cfg(_cfg)
    task_desc = load_task_desc(cfg)

    # Prepare the agent workspace
    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    
    # Initialize the journal and get metric information
    journal = Journal()
    metric_class = get_metric(config['competition']['name'])
    higher_is_better = metric_class().higher_is_better
    
    # Create MLE-STAR specific config
    mlestar_config_dict = config.get('mlestar', {})
    mlestar_cfg = MLEStarConfig(
        search_iterations=mlestar_config_dict.get('search_iterations', 3),
        refinement_iterations=mlestar_config_dict.get('refinement_iterations', 5),
        perplexity_model=mlestar_config_dict.get('perplexity_model', 'llama-3.1-sonar-large-128k-online'),
        enable_web_search=mlestar_config_dict.get('enable_web_search', True),
        enable_ablation=mlestar_config_dict.get('enable_ablation', True),
        enable_refinement=mlestar_config_dict.get('enable_refinement', True),
        enable_ensemble=mlestar_config_dict.get('enable_ensemble', True),
    )
    
    # Create the MLE-STAR agent
    agent = MLESTARAgent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        higher_is_better=higher_is_better,
        data_dir=os.path.join(cfg.workspace_dir, "input"),
        output_dir=cfg.workspace_dir,
        mlestar_cfg=mlestar_cfg,
    )
    
    logger.info(f"MLE-STAR agent set up for competition: {config['competition']['name']}")
    logger.info(f"Search iterations: {mlestar_cfg.search_iterations}, Refinement iterations: {mlestar_cfg.refinement_iterations}")
    return agent, journal, cfg

