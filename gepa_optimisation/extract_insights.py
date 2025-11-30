"""
GEPA Optimization Insights Extractor

This module extracts and formats key insights from GEPA optimization results:
- Best prompt found
- Reflection insights (LLM's analysis of what worked/didn't work)
- Reasoning behind mutations (why prompts were changed)
- Performance trends across iterations
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class GEPAInsightsExtractor:
    """Extract and format insights from GEPA optimization results."""
    
    def __init__(self, output_dir: str = "./gepa_insights"):
        """
        Initialize the insights extractor.
        
        Args:
            output_dir: Directory to save insight reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_and_save(self, result, competition_name: str = "unknown"):
        """
        Extract insights from GEPA optimization result and save to files.
        
        Args:
            result: The result object returned by gepa.optimize()
            competition_name: Name of the competition being optimized
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract all insights
        insights = {
            "metadata": {
                "competition": competition_name,
                "timestamp": timestamp,
                "optimization_complete": True
            },
            "best_prompt": self._extract_best_prompt(result),
            "best_score": self._extract_best_score(result),
            "reflection_insights": self._extract_reflection_insights(result),
            "mutation_reasoning": self._extract_mutation_reasoning(result),
            "performance_history": self._extract_performance_history(result),
            "optimization_summary": self._create_summary(result)
        }
        
        # Save to JSON
        json_path = self.output_dir / f"gepa_insights_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(insights, f, indent=2)
        print(f"✓ Saved insights to {json_path}")
        
        # Save human-readable report
        report_path = self.output_dir / f"gepa_report_{timestamp}.txt"
        self._save_human_readable_report(insights, report_path)
        print(f"✓ Saved human-readable report to {report_path}")
        
        # Print to console
        self._print_insights(insights)
        
        return insights
    
    def _extract_best_prompt(self, result) -> str:
        """Extract the best prompt from optimization result."""
        try:
            if hasattr(result, 'best_candidate'):
                if isinstance(result.best_candidate, dict):
                    return result.best_candidate.get('system_prompt', 'N/A')
                elif hasattr(result.best_candidate, 'system_prompt'):
                    return result.best_candidate.system_prompt
            return 'N/A'
        except Exception as e:
            print(f"Warning: Could not extract best prompt: {e}")
            return 'N/A'
    
    def _extract_best_score(self, result) -> float:
        """Extract the best score achieved."""
        try:
            if hasattr(result, 'best_score'):
                return float(result.best_score)
            return 0.0
        except Exception as e:
            print(f"Warning: Could not extract best score: {e}")
            return 0.0
    
    def _extract_reflection_insights(self, result) -> List[Dict[str, Any]]:
        """
        Extract reflection insights from GEPA's history.
        
        Reflection insights are the LLM's analysis of what worked and what didn't.
        This typically includes:
        - What improvements were observed
        - What mistakes were made
        - What patterns were identified
        """
        insights = []
        
        try:
            if hasattr(result, 'history') and result.history:
                for idx, entry in enumerate(result.history):
                    # GEPA stores reflection outputs in various formats
                    # Check for reflection data in the history entry
                    reflection_data = {}
                    
                    if isinstance(entry, dict):
                        # Look for reflection-related keys
                        reflection_data['iteration'] = idx
                        reflection_data['candidate'] = entry.get('candidate', {})
                        reflection_data['score'] = entry.get('score', 0.0)
                        
                        # Extract reflection text if available
                        if 'reflection' in entry:
                            reflection_data['reflection_text'] = entry['reflection']
                        elif 'analysis' in entry:
                            reflection_data['reflection_text'] = entry['analysis']
                        elif 'feedback' in entry:
                            reflection_data['reflection_text'] = entry['feedback']
                        
                        # Extract key insights from trajectories
                        if 'trajectories' in entry and entry['trajectories']:
                            reflection_data['trajectory_summary'] = self._summarize_trajectory(
                                entry['trajectories']
                            )
                        
                        insights.append(reflection_data)
                    elif hasattr(entry, '__dict__'):
                        # Handle object-based entries
                        reflection_data['iteration'] = idx
                        reflection_data['score'] = getattr(entry, 'score', 0.0)
                        
                        if hasattr(entry, 'reflection'):
                            reflection_data['reflection_text'] = entry.reflection
                        
                        insights.append(reflection_data)
            
            # If no insights found in history, note it
            if not insights:
                insights.append({
                    'note': 'No reflection insights found in result.history',
                    'suggestion': 'GEPA may not have stored reflection data, or format is different than expected'
                })
        
        except Exception as e:
            print(f"Warning: Error extracting reflection insights: {e}")
            insights.append({'error': str(e)})
        
        return insights
    
    def _extract_mutation_reasoning(self, result) -> List[Dict[str, Any]]:
        """
        Extract reasoning behind prompt mutations.
        
        This shows:
        - What changes were made to prompts
        - Why those changes were suggested
        - Before/after comparisons
        """
        mutations = []
        
        try:
            if hasattr(result, 'history') and result.history:
                for idx in range(1, len(result.history)):
                    prev_entry = result.history[idx - 1]
                    curr_entry = result.history[idx]
                    
                    mutation_data = {
                        'iteration': idx,
                        'from_score': self._get_score_from_entry(prev_entry),
                        'to_score': self._get_score_from_entry(curr_entry),
                    }
                    
                    # Extract prompt changes
                    prev_prompt = self._get_prompt_from_entry(prev_entry)
                    curr_prompt = self._get_prompt_from_entry(curr_entry)
                    
                    if prev_prompt and curr_prompt:
                        mutation_data['prompt_before'] = prev_prompt[:200] + "..."
                        mutation_data['prompt_after'] = curr_prompt[:200] + "..."
                        mutation_data['change_summary'] = self._summarize_prompt_change(
                            prev_prompt, curr_prompt
                        )
                    
                    # Look for mutation reasoning
                    if isinstance(curr_entry, dict):
                        if 'mutation_reason' in curr_entry:
                            mutation_data['reasoning'] = curr_entry['mutation_reason']
                        elif 'explanation' in curr_entry:
                            mutation_data['reasoning'] = curr_entry['explanation']
                    
                    mutations.append(mutation_data)
            
            if not mutations:
                mutations.append({
                    'note': 'No mutation data found in history',
                    'suggestion': 'GEPA may not store mutation reasoning, or format differs'
                })
        
        except Exception as e:
            print(f"Warning: Error extracting mutation reasoning: {e}")
            mutations.append({'error': str(e)})
        
        return mutations
    
    def _extract_performance_history(self, result) -> List[Dict[str, Any]]:
        """Extract performance metrics across all iterations."""
        history = []
        
        try:
            if hasattr(result, 'history') and result.history:
                for idx, entry in enumerate(result.history):
                    history_entry = {
                        'iteration': idx,
                        'score': self._get_score_from_entry(entry)
                    }
                    
                    # Add metadata if available
                    if isinstance(entry, dict):
                        if 'timestamp' in entry:
                            history_entry['timestamp'] = entry['timestamp']
                        if 'num_evaluations' in entry:
                            history_entry['evaluations'] = entry['num_evaluations']
                    
                    history.append(history_entry)
            
            # Calculate trend statistics
            if len(history) > 1:
                scores = [h['score'] for h in history if h['score'] > 0]
                if scores:
                    history.append({
                        'summary': {
                            'total_iterations': len(history) - 1,
                            'initial_score': scores[0],
                            'final_score': scores[-1],
                            'improvement': scores[-1] - scores[0],
                            'improvement_pct': ((scores[-1] - scores[0]) / scores[0] * 100) if scores[0] > 0 else 0,
                            'best_score': max(scores),
                            'avg_score': sum(scores) / len(scores)
                        }
                    })
        
        except Exception as e:
            print(f"Warning: Error extracting performance history: {e}")
        
        return history
    
    def _create_summary(self, result) -> Dict[str, Any]:
        """Create a high-level summary of the optimization."""
        summary = {
            'optimization_successful': hasattr(result, 'best_score') and result.best_score > 0,
            'total_iterations': 0,
            'key_findings': []
        }
        
        try:
            if hasattr(result, 'history') and result.history:
                summary['total_iterations'] = len(result.history)
                
                # Identify key findings from history
                scores = [self._get_score_from_entry(e) for e in result.history]
                if scores:
                    # Find biggest improvement
                    improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
                    if improvements:
                        best_improvement_idx = improvements.index(max(improvements)) + 1
                        summary['key_findings'].append({
                            'finding': 'Biggest improvement',
                            'iteration': best_improvement_idx,
                            'improvement': max(improvements),
                            'score_before': scores[best_improvement_idx - 1],
                            'score_after': scores[best_improvement_idx]
                        })
                    
                    # Check if optimization converged
                    if len(scores) >= 3:
                        recent_scores = scores[-3:]
                        score_variance = max(recent_scores) - min(recent_scores)
                        summary['convergence'] = {
                            'converged': score_variance < 0.01,
                            'final_variance': score_variance
                        }
        
        except Exception as e:
            print(f"Warning: Error creating summary: {e}")
        
        return summary
    
    def _get_score_from_entry(self, entry) -> float:
        """Extract score from a history entry."""
        if isinstance(entry, dict):
            return entry.get('score', 0.0)
        elif hasattr(entry, 'score'):
            return entry.score
        return 0.0
    
    def _get_prompt_from_entry(self, entry) -> str:
        """Extract prompt from a history entry."""
        if isinstance(entry, dict):
            candidate = entry.get('candidate', {})
            if isinstance(candidate, dict):
                return candidate.get('system_prompt', '')
            return ''
        elif hasattr(entry, 'candidate'):
            if isinstance(entry.candidate, dict):
                return entry.candidate.get('system_prompt', '')
            elif hasattr(entry.candidate, 'system_prompt'):
                return entry.candidate.system_prompt
        return ''
    
    def _summarize_trajectory(self, trajectories: List[str]) -> str:
        """Summarize agent trajectories."""
        if not trajectories:
            return "No trajectories available"
        
        # Simple summary: count steps, check for errors
        total_steps = sum(traj.count('--- Step') for traj in trajectories)
        has_errors = any('error' in traj.lower() or 'failed' in traj.lower() 
                        for traj in trajectories)
        
        return f"{total_steps} total steps, {'with' if has_errors else 'no'} errors"
    
    def _summarize_prompt_change(self, before: str, after: str) -> str:
        """Summarize what changed between two prompts."""
        if before == after:
            return "No change"
        
        # Simple diff: check length and key phrase changes
        length_change = len(after) - len(before)
        summary = f"Length {'increased' if length_change > 0 else 'decreased'} by {abs(length_change)} chars"
        
        # Check for key phrase additions/removals
        before_lower = before.lower()
        after_lower = after.lower()
        
        key_phrases = ['cross-validation', 'feature engineering', 'submission', 
                      'validation', 'accuracy', 'model', 'train']
        
        added = [phrase for phrase in key_phrases if phrase not in before_lower and phrase in after_lower]
        removed = [phrase for phrase in key_phrases if phrase in before_lower and phrase not in after_lower]
        
        if added:
            summary += f", added: {', '.join(added)}"
        if removed:
            summary += f", removed: {', '.join(removed)}"
        
        return summary
    
    def _save_human_readable_report(self, insights: Dict, path: Path):
        """Save a human-readable text report."""
        with open(path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GEPA OPTIMIZATION INSIGHTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Competition: {insights['metadata']['competition']}\n")
            f.write(f"Timestamp: {insights['metadata']['timestamp']}\n\n")
            
            # Best Results
            f.write("-" * 80 + "\n")
            f.write("BEST RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Score: {insights['best_score']:.4f}\n\n")
            f.write("Best Prompt:\n")
            f.write(insights['best_prompt'] + "\n\n")
            
            # Optimization Summary
            f.write("-" * 80 + "\n")
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = insights['optimization_summary']
            f.write(f"Total Iterations: {summary.get('total_iterations', 'N/A')}\n")
            f.write(f"Optimization Successful: {summary.get('optimization_successful', False)}\n\n")
            
            if 'key_findings' in summary and summary['key_findings']:
                f.write("Key Findings:\n")
                for finding in summary['key_findings']:
                    f.write(f"  - {finding.get('finding', 'N/A')}\n")
                    f.write(f"    Iteration: {finding.get('iteration', 'N/A')}\n")
                    f.write(f"    Improvement: {finding.get('improvement', 0):.4f}\n\n")
            
            # Reflection Insights
            f.write("-" * 80 + "\n")
            f.write("REFLECTION INSIGHTS\n")
            f.write("-" * 80 + "\n")
            for idx, insight in enumerate(insights['reflection_insights']):
                f.write(f"\nIteration {insight.get('iteration', idx)}:\n")
                f.write(f"  Score: {insight.get('score', 'N/A')}\n")
                if 'reflection_text' in insight:
                    f.write(f"  Reflection: {insight['reflection_text']}\n")
                if 'trajectory_summary' in insight:
                    f.write(f"  Trajectory: {insight['trajectory_summary']}\n")
            
            # Mutation Reasoning
            f.write("\n" + "-" * 80 + "\n")
            f.write("MUTATION REASONING\n")
            f.write("-" * 80 + "\n")
            for mutation in insights['mutation_reasoning']:
                f.write(f"\nIteration {mutation.get('iteration', 'N/A')}:\n")
                f.write(f"  Score Change: {mutation.get('from_score', 0):.4f} → {mutation.get('to_score', 0):.4f}\n")
                if 'reasoning' in mutation:
                    f.write(f"  Reasoning: {mutation['reasoning']}\n")
                if 'change_summary' in mutation:
                    f.write(f"  Changes: {mutation['change_summary']}\n")
            
            # Performance History
            f.write("\n" + "-" * 80 + "\n")
            f.write("PERFORMANCE HISTORY\n")
            f.write("-" * 80 + "\n")
            for entry in insights['performance_history']:
                if 'summary' in entry:
                    f.write("\nSummary Statistics:\n")
                    for key, value in entry['summary'].items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"Iteration {entry.get('iteration', 'N/A')}: {entry.get('score', 'N/A'):.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def _print_insights(self, insights: Dict):
        """Print key insights to console."""
        print("\n" + "=" * 80)
        print("GEPA OPTIMIZATION INSIGHTS")
        print("=" * 80)
        
        print(f"\n📊 Competition: {insights['metadata']['competition']}")
        print(f"🏆 Best Score: {insights['best_score']:.4f}")
        print(f"🔁 Total Iterations: {insights['optimization_summary'].get('total_iterations', 'N/A')}")
        
        print("\n" + "-" * 80)
        print("✨ BEST PROMPT FOUND:")
        print("-" * 80)
        print(insights['best_prompt'])
        
        print("\n" + "-" * 80)
        print("🔍 KEY REFLECTION INSIGHTS:")
        print("-" * 80)
        for insight in insights['reflection_insights'][:3]:  # Show first 3
            print(f"\nIteration {insight.get('iteration', 'N/A')} (Score: {insight.get('score', 'N/A'):.4f}):")
            if 'reflection_text' in insight:
                print(f"  {insight['reflection_text'][:200]}...")
            elif 'note' in insight:
                print(f"  {insight['note']}")
        
        print("\n" + "-" * 80)
        print("🔄 MUTATION REASONING:")
        print("-" * 80)
        for mutation in insights['mutation_reasoning'][:3]:  # Show first 3
            print(f"\nIteration {mutation.get('iteration', 'N/A')}:")
            print(f"  Score: {mutation.get('from_score', 0):.4f} → {mutation.get('to_score', 0):.4f}")
            if 'reasoning' in mutation:
                print(f"  Why: {mutation['reasoning'][:150]}...")
            if 'change_summary' in mutation:
                print(f"  Changes: {mutation['change_summary']}")
        
        print("\n" + "=" * 80)
