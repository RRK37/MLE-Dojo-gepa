"""
Setup module for MLE-STAR Agent.

This module provides functions to set up and configure the MLE-STAR Agent
for Kaggle competitions, following the AIDE pattern for MLE-Dojo integration.
"""

import os
import logging
from typing import Dict, Any, Tuple

from mledojo.agent.mlestar.agent import MLEStarAgent, MLEStarConfig
from mledojo.agent.aide.journal import Journal
from mledojo.agent.aide.utils.config import (
    _load_cfg,
    prep_cfg,
    load_task_desc,
    prep_agent_workspace,
)
from mledojo.utils import get_metric

logger = logging.getLogger("mlestar_setup")


def setup_mlestar_agent(
    config: Dict[str, Any],
) -> Tuple[MLEStarAgent, Journal, Any]:
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
    # Load and prepare AIDE-style config (for workspace management)
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
    logger.info("Preparing agent workspace (copying and extracting files)...")
    prep_agent_workspace(cfg)
    
    # Initialize the journal and get metric information
    journal = Journal()
    metric_class = get_metric(config['competition']['name'])
    higher_is_better = metric_class().higher_is_better
    
    # Load MLE-STAR specific config
    mlestar_cfg = MLEStarConfig()
    if 'mlestar' in config:
        mlestar_cfg = MLEStarConfig.from_dict(config['mlestar'])
    
    # Create the agent
    agent = MLEStarAgent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        higher_is_better=higher_is_better,
        data_dir=os.path.join(cfg.workspace_dir, "input"),
        output_dir=cfg.workspace_dir,
        mlestar_cfg=mlestar_cfg,
    )
    
    logger.info(f"MLE-STAR agent set up for competition: {config['competition']['name']}")
    return agent, journal, cfg
