"""
GEPA Adapter for MLE-Dojo AIDE Agent

This module implements the GEPAAdapter protocol to enable GEPA optimization
of AIDE agent prompts based on Kaggle competition performance.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Mapping, Sequence, Optional
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
    max_steps: int = 5
    execution_timeout: int = 3600
    output_dir: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"./gepa_temp_outputs/{self.name}"


@dataclass  
class ExecutionTrajectory:
    """Trajectory data captured from AIDE agent run"""
    journal_export: Dict
    conversation_history: List[Dict]
    cost_history: List[Dict]
    final_score: float
    best_score: float
    num_good_nodes: int
    num_buggy_nodes: int
    failure_patterns: List[str]


@dataclass
class GEPAEvaluationResult:
    """
    Return object required by GEPA's evaluator.
    Must have 'outputs' and 'scores' attributes.
    """
    outputs: List[Any]
    scores: List[float]
    trajectories: Optional[List[ExecutionTrajectory]] = None


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
        self.base_config = base_config
        self.verbose = verbose
        self.run_counter = 0
        
    def evaluate(
        self,
        batch: List[CompetitionConfig],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> GEPAEvaluationResult:
        """
        Evaluate a candidate prompt configuration on a batch of competitions.
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
                # On failure, assign score of 0
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
        
        # FIX: Return the specific object GEPA expects, not a dict
        return GEPAEvaluationResult(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )
    
    def _run_aide_agent(
        self,
        comp_config: CompetitionConfig,
        custom_prompts: Dict[str, str],
        capture_trace: bool,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Run AIDE agent with custom prompts on a single competition.
        """
        # Prepare configuration
        config = self._prepare_config(comp_config)
        
        # Determine competition data path
        comp_data_path = os.path.join(comp_config.data_dir, comp_config.name, "data")
        
        # Get metric class
        try:
            metric_class = get_metric(comp_config.name)
            higher_is_better = metric_class().higher_is_better
        except Exception as e:
            logger.warning(f"Could not load metric for {comp_config.name}, defaulting to True: {e}")
            metric_class = None
            higher_is_better = True
        
        registry = CompetitionRegistry(
            name=comp_config.name,
            data_dir=comp_data_path,
            comp_info=CompInfo(
                category="Tabular",
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
            render_mode=config['env'].get('render_mode', None)
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
        
        # FIX: Dynamically set the description file path
        # Otherwise it uses the hardcoded path from base config (e.g. Titanic)
        desc_path = os.path.join(comp_config.data_dir, comp_config.name, "data", "public", "description.txt")
        
        if os.path.exists(desc_path):
            config['desc_file'] = desc_path
        else:
            # Clear it so it doesn't default to the wrong competition
            config['desc_file'] = None
            logger.warning(f"Description file not found at {desc_path}")
            
        return config
    
    def _build_trajectory(
        self,
        journal: Journal,
        agent: Agent,
        final_score: float
    ) -> ExecutionTrajectory:
        
        failure_patterns = []
        for node in journal.buggy_nodes:
            if node.feedback:
                err_msg = str(node.feedback.get('error', 'Unknown error'))
                failure_patterns.append(err_msg[:200])
        
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
        eval_batch: GEPAEvaluationResult, # Updated type hint
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build reflective dataset for GEPA's instruction proposer.
        """
        # Access attributes directly now
        trajectories = eval_batch.trajectories
        scores = eval_batch.scores
        
        if not trajectories:
            return {comp: [] for comp in components_to_update}
        
        reflective_dataset = {}
        
        for component in components_to_update:
            examples = []
            
            for traj, score in zip(trajectories, scores):
                if score < 0.5:
                    example = self._build_reflective_example(
                        component=component,
                        trajectory=traj,
                        score=score,
                        current_prompt=candidate.get(component, "")
                    )
                    examples.append(example)
            
            reflective_dataset[component] = examples[:10]
        
        return reflective_dataset
    
    def _build_reflective_example(
        self,
        component: str,
        trajectory: ExecutionTrajectory,
        score: float,
        current_prompt: str
    ) -> Dict[str, Any]:
        
        if 'draft' in component:
            stage = 'drafting'
        elif 'improve' in component:
            stage = 'improving'
        elif 'debug' in component:
            stage = 'debugging'
        else:
            stage = 'unknown'
        
        feedback_parts = []
        
        if trajectory.failure_patterns:
            feedback_parts.append(
                f"Common errors: {'; '.join(trajectory.failure_patterns[:3])}"
            )
        
        feedback_parts.append(
            f"Achieved only {trajectory.num_good_nodes} good solutions out of "
            f"{trajectory.num_good_nodes + trajectory.num_buggy_nodes} attempts"
        )
        
        feedback_parts.append(f"Final score: {score:.4f}")
        
        return {
            "Inputs": {
                "stage": stage,
                "current_prompt": current_prompt[:500],
            },
            "Generated Outputs": {
                "num_good_nodes": trajectory.num_good_nodes,
                "num_buggy_nodes": trajectory.num_buggy_nodes,
            },
            "Feedback": " | ".join(feedback_parts),
            "Score": score
        }