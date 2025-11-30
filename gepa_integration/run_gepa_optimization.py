"""
Main orchestration script for GEPA optimization of AIDE agent prompts.

Usage:
    python run_gepa_optimization.py --config gepa_integration/config_gepa.yaml
    python run_gepa_optimization.py --competition titanic --max-iterations 5
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add gepa_integration to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from gepa_integration.mledojo_gepa_adapter import MLEDojoGEPAAdapter, CompetitionConfig
from gepa_integration.prompt_utils import extract_default_prompts, serialize_prompts
from gepa_integration.gepa_logger import GEPALogger
from mledojo.utils import load_config

# Try importing GEPA
try:
    import gepa
    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False
    print("WARNING: GEPA not installed. Install with: pip install gepa")

logger = logging.getLogger("gepa_optimization")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization on AIDE agent prompts"
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        default='gepa_integration/config_gepa.yaml',
        help='Path to GEPA configuration file'
    )
    
    parser.add_argument(
        '--competition',
        type=str,
        help='Single competition to run (overrides config)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        help='Maximum GEPA iterations (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--base-config',
        type=str,
        default='config.yaml',
        help='Path to base MLE-Dojo config.yaml'
    )
    
    return parser.parse_args()


def load_gepa_config(config_path: str) -> dict:
    """Load GEPA configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO"):
    """Setup basic logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_competition_datasets(
    gepa_config: dict,
    data_base_dir: str
) -> tuple:
    """
    Create train and validation datasets from competitions.
    
    Returns:
        (trainset, valset) where each is a list of CompetitionConfig
    """
    train_comps = gepa_config['gepa']['train_competitions']
    val_comps = gepa_config['gepa']['val_competitions']
    
    trainset = [
        CompetitionConfig(
            name=comp,
            data_dir=data_base_dir,
            max_steps=gepa_config['aide']['max_steps'],
            execution_timeout=gepa_config['aide']['execution_timeout']
        )
        for comp in train_comps
    ]
    
    valset = [
        CompetitionConfig(
            name=comp,
            data_dir=data_base_dir,
            max_steps=gepa_config['aide']['max_steps'],
            execution_timeout=gepa_config['aide']['execution_timeout']
        )
        for comp in val_comps
    ]
    
    return trainset, valset


def main():
    """Main orchestration function"""
    if not GEPA_AVAILABLE:
        logger.error("GEPA is not installed. Cannot proceed.")
        logger.error("Install with: pip install gepa")
        return 1
    
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    gepa_config = load_gepa_config(args.config)
    base_config = load_config(args.base_config)
    
    # Override config with command line args if provided
    if args.max_iterations:
        gepa_config['gepa']['max_iterations'] = args.max_iterations
    
    if args.competition:
        gepa_config['gepa']['train_competitions'] = [args.competition]
        gepa_config['gepa']['val_competitions'] = [args.competition]
    
    # Setup logging
    setup_logging(gepa_config['output'].get('log_level', 'INFO'))
    
    # Create output directory
    output_base = args.output_dir or gepa_config['output']['base_dir']
    output_dir = Path(output_base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize GEPA logger
    gepa_logger = GEPALogger(log_dir=output_dir / "logs")
    
    # Create datasets
    data_base_dir = gepa_config['data']['base_dir']
    trainset, valset = create_competition_datasets(gepa_config, data_base_dir)
    
    logger.info(f"Train competitions: {[c.name for c in trainset]}")
    logger.info(f"Val competitions: {[c.name for c in valset]}")
    
    # Extract seed candidate (default prompts)
    seed_candidate = extract_default_prompts()
    logger.info("Using default AIDE prompts as seed candidate")
    
    # Save seed candidate
    serialize_prompts(seed_candidate, output_dir / "seed_prompts.json")
    
    # Initialize adapter
    adapter = MLEDojoGEPAAdapter(
        base_config=base_config,
        verbose=True
    )
    
    logger.info("Starting GEPA optimization...")
    logger.info(f"Max iterations: {gepa_config['gepa']['max_iterations']}")
    logger.info(f"Reflection LM: {gepa_config['gepa']['reflection_lm']}")
    logger.info(f"Task LM: {gepa_config['gepa']['task_lm']}")
    
    try:
        # Run GEPA optimization
        # Note: This is a simplified version. Full GEPA integration would require
        # implementing the full GEPAAdapter protocol methods as async or with proper
        # evaluation functions
        
        logger.warning("GEPA optimization loop not fully implemented yet.")
        logger.warning("This is a template. You need to:")
        logger.warning("1. Call gepa.optimize() with proper parameters")
        logger.warning("2. Implement evaluation function that calls adapter.evaluate()")
        logger.warning("3. Handle iteration logging with gepa_logger")
        logger.warning("4. Save best candidate prompts")
        
        # Placeholder for gepa.optimize call:
        result = gepa.optimize(
             seed_candidate=seed_candidate,
             trainset=trainset,
             valset=valset,
             task_lm=gepa_config['gepa']['task_lm'],
             reflection_lm=gepa_config['gepa']['reflection_lm'],
             max_metric_calls=gepa_config['gepa']['max_iterations'],
             adapter=adapter
        )
        
        # For now, just demonstrate the adapter works
        #logger.info("Testing adapter with seed candidate...")
        #test_batch = trainset[:1]  # Test with one competition
        
        #eval_result = adapter.evaluate(
        #    batch=test_batch,
        #    candidate=seed_candidate,
        #    capture_traces=True
        #)
        
        logger.info(f"Test evaluation completed:")
        logger.info(f"  - Outputs: {len(eval_result['outputs'])}")
        logger.info(f"  - Scores: {eval_result['scores']}")
        logger.info(f"  - Trajectories: {len(eval_result['trajectories']) if eval_result['trajectories'] else 0}")
        
        # Save test results
        import json
        with open(output_dir / "test_evaluation_result.json", 'w') as f:
            # Don't serialize trajectories (too large)
            result_summary = {
                'scores': eval_result['scores'],
                'outputs': eval_result['outputs']
            }
            json.dump(result_summary, f, indent=2)
        
        logger.info(f"Test results saved to {output_dir / 'test_evaluation_result.json'}")
        
        # Generate plots and reports
        if gepa_config['output'].get('generate_plots', True):
            gepa_logger.plot_optimization_progress()
        
        gepa_logger.save_summary_report()
        
        logger.info(f"Optimization complete! Results in: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
