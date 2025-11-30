"""
Enhanced logging utilities for GEPA optimization.

This module provides structured logging for tracking GEPA optimization progress,
including per-run metrics, prompt evolution, and visualization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt


class GEPALogger:
    """Logger for GEPA optimization runs"""
    
    def __init__(self, log_dir: str | Path, experiment_name: str = None):
        """
        Initialize GEPA logger.
        
        Args:
            log_dir: Directory for saving logs
            experiment_name: Name for this optimization run
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"gepa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.run_log_file = self.log_dir / f"{experiment_name}_runs.jsonl"
        self.iteration_log_file = self.log_dir / f"{experiment_name}_iterations.jsonl"
        
        # Setup Python logger
        self.logger = logging.getLogger(f"gepa_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / f"{experiment_name}.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_run(self, run_data: Dict[str, Any]) -> None:
        """
        Log a single AIDE agent run.
        
        Args:
            run_data: Dict containing run information
        """
        run_data['timestamp'] = datetime.now().isoformat()
        
        with open(self.run_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(run_data) + '\n')
        
        self.logger.info(f"Run logged: {run_data.get('run_id', 'unknown')}")
    
    def log_iteration(self, iteration_data: Dict[str, Any]) -> None:
        """
        Log a GEPA optimization iteration.
        
        Args:
            iteration_data: Dict containing iteration metrics and prompts
        """
        iteration_data['timestamp'] = datetime.now().isoformat()
        
        with open(self.iteration_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(iteration_data) + '\n')
        
        iter_num = iteration_data.get('iteration', '?')
        score = iteration_data.get('validation_score', 0)
        self.logger.info(f"Iteration {iter_num}: validation score = {score:.4f}")
    
    def load_iteration_history(self) -> List[Dict]:
        """Load all iteration logs"""
        if not self.iteration_log_file.exists():
            return []
        
        history = []
        with open(self.iteration_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                history.append(json.loads(line.strip()))
        
        return history
    
    def plot_optimization_progress(self, save_path: str | Path = None) -> None:
        """
        Plot GEPA optimization progress over iterations.
        
        Args:
            save_path: Optional path to save plot image
        """
        history = self.load_iteration_history()
        
        if not history:
            self.logger.warning("No iteration history to plot")
            return
        
        iterations = [h['iteration'] for h in history]
        train_scores = [h.get('train_score', 0) for h in history]
        val_scores = [h.get('validation_score', 0) for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_scores, 'o-', label='Train Score', alpha=0.7)
        plt.plot(iterations, val_scores, 's-', label='Validation Score', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title(f'GEPA Optimization Progress - {self.experiment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved progress plot to {save_path}")
        else:
            default_path = self.log_dir / f"{self.experiment_name}_progress.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved progress plot to {default_path}")
        
        plt.close()
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the optimization.
        
        Returns:
            Markdown-formatted summary report
        """
        history = self.load_iteration_history()
        
        if not history:
            return "No optimization history available."
        
        best_iter = max(history, key=lambda x: x.get('validation_score', 0))
        
        report = [
            f"# GEPA Optimization Summary: {self.experiment_name}",
            "",
            f"**Total Iterations:** {len(history)}",
            f"**Best Validation Score:** {best_iter.get('validation_score', 0):.4f}",
            f"**Best Iteration:** {best_iter.get('iteration', '?')}",
            "",
            "## Optimization Trajectory",
            ""
        ]
        
        for h in history:
            report.append(
                f"- Iteration {h['iteration']}: "
                f"train={h.get('train_score', 0):.4f}, "
                f"val={h.get('validation_score', 0):.4f}"
            )
        
        report.append("")
        report.append("## Best Prompts")
        report.append("")
        
        if 'candidate' in best_iter:
            for key, value in best_iter['candidate'].items():
                report.append(f"####{key}")
                report.append(f"```\n{value}\n```")
                report.append("")
        
        return "\n".join(report)
    
    def save_summary_report(self, filepath: str | Path = None) -> None:
        """Save summary report to file"""
        if filepath is None:
            filepath = self.log_dir / f"{self.experiment_name}_summary.md"
        
        report = self.generate_summary_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Saved summary report to {filepath}")
