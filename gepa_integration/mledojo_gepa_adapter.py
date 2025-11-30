"""
GEPA Adapter for MLE-Dojo AIDE Agent

This module implements the GEPAAdapter protocol to enable GEPA optimization
of AIDE agent prompts based on Kaggle competition performance.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mledojo.agent.aide.buildup import setup_aide_agent
from mledojo.agent.aide.agent import Agent
from mledojo.agent.aide.journal import Journal
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.utils import load_config, create_config_from_args, get_metric

logger = logging.getLogger("gepa_mledojo")


@dataclass
class CompetitionConfig:
    """Configuration for a single competition run (DataInst type)"""
    name: str
    data_dir: str
    max_steps: int = 5  # Reduced for faster GEPA iterations
    execution_timeout: int = 3600
    output_dir: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"./gepa_temp_outputs/{self.name}"


@dataclass  
class ExecutionTrajectory:
    """Trajectory data captured from AIDE agent run"""
    journal_export: Dict  # Full journal serialized
    conversation_history: List[Dict]
    cost_history: List[Dict]
    final_score: float
    best_score: float
    num_good_nodes: int
    num_buggy_nodes: int
    failure_patterns: List[str]
    
    
class MLEDojoGEPAAdapter:
    """
    GEPA Adapter for MLE-Dojo AIDE Agent.
    
    Implements the GEPAAdapter protocol to optimize AIDE agent prompts.
    """
    
    def __init__(
        self, 
        base_config: Dict[str, Any],
        verbose: bool = True
    ):
        """
        Initialize the adapter.
        
        Args:
            base_config: Base configuration for AIDE agent (from config.yaml)
            verbose: Whether to log detailed information
        """
        self.base_config = base_config
        self.verbose = verbose
        self.run_counter = 0
        
    def evaluate(
        self,
        batch: List[CompetitionConfig],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate prompt configuration on a batch of competitions.
        
        Args:
            batch: List of competition configurations to run
            candidate: Dict mapping component names to prompt text
                      Expected keys: 'introduction_draft', 'introduction_improve', 'introduction_debug'
            capture_traces: Whether to capture detailed execution traces
            
        Returns:
            EvaluationBatch-like dict with:
                - outputs: List of final submission results
                - scores: List of position scores (higher is better)
                - trajectories: List of ExecutionTrajectory objects (if capture_traces=True)
        """
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None
        
        logger.info(f"Evaluating candidate on {len(batch)} competitions")
        
        for comp_config in batch:
            self.run_counter += 1
            run_id = f"run_{self.run_counter:04d}"
            
            try:
                result = self._run_aide_agent(
                    comp_config=comp_config,
                    custom_prompts=candidate,
                    capture_trace=capture_traces,
                    run_id=run_id
                )
                
                outputs.append(result['output'])
                scores.append(result['score'])
                
                if capture_traces:
                    trajectories.append(result['trajectory'])
                    
                logger.info(f"{run_id} | {comp_config.name} | Score: {result['score']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {comp_config.name}: {e}")
                # On failure, assign score of 0 and capture error
                outputs.append({'error': str(e)})
                scores.append(0.0)
                
                if capture_traces:
                    trajectories.append(ExecutionTrajectory(
                        journal_export={},
                        conversation_history=[],
                        cost_history=[],
                        final_score=0.0,
                        best_score=0.0,
                        num_good_nodes=0,
                        num_buggy_nodes=0,
                        failure_patterns=[str(e)]
                    ))
        
        return {
            'outputs': outputs,
            'scores': scores,
            'trajectories': trajectories
        }
    
    def _run_aide_agent(
        self,
        comp_config: CompetitionConfig,
        custom_prompts: Dict[str, str],
        capture_trace: bool,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Run AIDE agent with custom prompts on a single competition.
        
        Returns dict with 'output', 'score', and optionally 'trajectory'
        """
        # Prepare configuration
        config = self._prepare_config(comp_config)
        
        # Determine competition data path
        # Assumes structure: <data_dir>/<competition_name>/data/
        comp_data_path = os.path.join(comp_config.data_dir, comp_config.name, "data")
        
        # Get metric class and determine higher_is_better
        metric_class = get_metric(comp_config.name)
        higher_is_better = metric_class().higher_is_better
        
        # Create registry and register the competition
        # This is required because KaggleEnvironment expects a registry instance
        registry = CompetitionRegistry(
            name=comp_config.name,
            data_dir=comp_data_path,
            comp_info=CompInfo(
                category="Tabular",  # Defaulting, could be inferred if needed
                level="intermediate",
                output_type="submission.csv",
                higher_is_better=higher_is_better
            ),
            metric_class=metric_class
        )
        
        # Setup environment
        env = KaggleEnvironment(
            competition_name=comp_config.name,
            output_dir=config['output_dir'],
            competition_registry=registry,
            render_mode=config['env']['render_mode']
        )
        
        # Setup AIDE agent with custom prompts
        agent, journal, cfg = setup_aide_agent(
            config=config,
            custom_prompts=custom_prompts
        )
        
        # Run the agent
        for step in range(comp_config.max_steps):
            try:
                def exec_callback(code: str) -> Dict:
                    """Execute code in environment"""
                    obs, reward = env.step("execute_code", **{"code": code})
                    return obs
                
                agent.step(exec_callback=exec_callback)
            except Exception as e:
                logger.warning(f"Step {step} failed: {e}")
                break
        
        # Extract results
        best_node = journal.get_best_node(only_good=False)
        best_score = best_node.position_score if best_node and best_node.position_score else 0.0
        
        output = {
            'competition': comp_config.name,
            'num_nodes': len(journal.nodes),
            'num_good_nodes': len(journal.good_nodes),
            'best_score': best_score
        }
        
        trajectory = None
        if capture_trace:
            trajectory = self._build_trajectory(journal, agent, best_score)
        
        return {
            'output': output,
            'score': best_score,
            'trajectory': trajectory
        }
    
    def _prepare_config(self, comp_config: CompetitionConfig) -> Dict[str, Any]:
        """Prepare configuration dict for AIDE agent"""
        config = self.base_config.copy()
        
        # Update with competition-specific settings
        config['competition']['name'] = comp_config.name
        config['competition']['data_dir'] = comp_config.data_dir
        config['env']['max_steps'] = comp_config.max_steps
        config['env']['execution_timeout'] = comp_config.execution_timeout
        config['output_dir'] = comp_config.output_dir
        
        return config
    
    def _build_trajectory(
        self,
        journal: Journal,
        agent: Agent,
        final_score: float
    ) -> ExecutionTrajectory:
        """Build trajectory object from journal and agent state"""
        
        # Get failure patterns
        failure_patterns = []
        for node in journal.buggy_nodes:
            if node.feedback:
                err_msg = str(node.feedback.get('error', 'Unknown error'))
                failure_patterns.append(err_msg[:200])  # Truncate long errors
        
        return ExecutionTrajectory(
            journal_export=journal.export_for_gepa(),
            conversation_history=agent.conversation_history.copy(),
            cost_history=agent.cost_history.copy(),
            final_score=final_score,
            best_score=final_score,
            num_good_nodes=len(journal.good_nodes),
            num_buggy_nodes=len(journal.buggy_nodes),
            failure_patterns=failure_patterns
        )
    
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: Dict[str, Any],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build reflective dataset for GEPA's instruction proposer.
        
        Args:
            candidate: The prompts that were evaluated
            eval_batch: Result from evaluate() with trajectories
            components_to_update: Which prompt components to optimize
            
        Returns:
            Dict mapping component name to list of reflective examples
        """
        trajectories = eval_batch['trajectories']
        scores = eval_batch['scores']
        
        if not trajectories:
            return {comp: [] for comp in components_to_update}
        
        reflective_dataset = {}
        
        for component in components_to_update:
            examples = []
            
            for traj, score in zip(trajectories, scores):
                # Focus on failures and low-scoring runs for reflection
                if score < 0.5:  # Threshold for "low performance"
                    example = self._build_reflective_example(
                        component=component,
                        trajectory=traj,
                        score=score,
                        current_prompt=candidate.get(component, "")
                    )
                    examples.append(example)
            
            reflective_dataset[component] = examples[:10]  # Limit to top 10 failures
        
        return reflective_dataset
    
    def _build_reflective_example(
        self,
        component: str,
        trajectory: ExecutionTrajectory,
        score: float,
        current_prompt: str
    ) -> Dict[str, Any]:
        """Build a single reflective example for a component"""
        
        # Extract relevant information based on component type
        if 'draft' in component:
            stage = 'drafting'
        elif 'improve' in component:
            stage = 'improving'
        elif 'debug' in component:
            stage = 'debugging'
        else:
            stage = 'unknown'
        
        feedback_parts = []
        
        # Add failure patterns
        if trajectory.failure_patterns:
            feedback_parts.append(
                f"Common errors: {'; '.join(trajectory.failure_patterns[:3])}"
            )
        
        # Add performance metrics
        feedback_parts.append(
            f"Achieved only {trajectory.num_good_nodes} good solutions out of "
            f"{trajectory.num_good_nodes + trajectory.num_buggy_nodes} attempts"
        )
        
        feedback_parts.append(f"Final score: {score:.4f}")
        
        return {
            "Inputs": {
                "stage": stage,
                "current_prompt": current_prompt[:500],  # Truncate for brevity
            },
            "Generated Outputs": {
                "num_good_nodes": trajectory.num_good_nodes,
                "num_buggy_nodes": trajectory.num_buggy_nodes,
            },
            "Feedback": " | ".join(feedback_parts),
            "Score": score
        }
