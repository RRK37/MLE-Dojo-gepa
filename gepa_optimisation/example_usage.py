"""
Example: How to use GEPA insights extraction

This script demonstrates different ways to extract and use optimization insights.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from gepa_optimisation.extract_insights import GEPAInsightsExtractor
from gepa_optimisation.analyze_results import (
    analyze_gepa_result,
    inspect_result_structure,
    save_result_to_file,
    load_result_from_file
)


def example_1_basic_extraction():
    """
    Example 1: Basic insights extraction after optimization
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Insights Extraction")
    print("="*80)
    
    # Simulated result object (replace with actual result from optimize())
    class MockResult:
        def __init__(self):
            self.best_score = 0.8342
            self.best_candidate = {
                'system_prompt': 'You are a Kaggle Grandmaster. Focus on feature engineering.'
            }
            self.history = [
                {
                    'iteration': 0,
                    'score': 0.7856,
                    'candidate': {'system_prompt': 'Initial prompt...'},
                    'reflection': 'Baseline established, need better feature engineering'
                },
                {
                    'iteration': 1,
                    'score': 0.8234,
                    'candidate': {'system_prompt': 'Improved prompt with CV...'},
                    'reflection': 'Cross-validation helped, but still missing key features',
                    'mutation_reason': 'Added explicit CV instructions'
                },
                {
                    'iteration': 2,
                    'score': 0.8342,
                    'candidate': {'system_prompt': 'Final prompt...'},
                    'reflection': 'Feature engineering significantly improved results',
                    'mutation_reason': 'Emphasized feature engineering and submission format'
                }
            ]
    
    # Create mock result
    result = MockResult()
    
    # Extract insights
    extractor = GEPAInsightsExtractor(output_dir="./example_insights")
    insights = extractor.extract_and_save(result, competition_name="titanic")
    
    print("\n✓ Insights extracted successfully!")
    print(f"  Check ./example_insights/ for output files")


def example_2_access_specific_insights():
    """
    Example 2: Access specific insight fields programmatically
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Accessing Specific Insights")
    print("="*80)
    
    # Load insights from JSON file (after running example 1)
    import json
    from pathlib import Path
    
    insights_dir = Path("./example_insights")
    json_files = list(insights_dir.glob("gepa_insights_*.json"))
    
    if not json_files:
        print("No insights files found. Run example_1_basic_extraction() first.")
        return
    
    with open(json_files[0]) as f:
        insights = json.load(f)
    
    # Access best prompt
    print("\n1. Best Prompt:")
    print(f"   {insights['best_prompt'][:100]}...")
    
    # Access best score
    print(f"\n2. Best Score: {insights['best_score']:.4f}")
    
    # Access reflection insights
    print("\n3. Reflection Insights:")
    for reflection in insights['reflection_insights'][:2]:
        print(f"   Iteration {reflection.get('iteration', 'N/A')}:")
        print(f"   Score: {reflection.get('score', 'N/A'):.4f}")
        if 'reflection_text' in reflection:
            print(f"   Insight: {reflection['reflection_text'][:80]}...")
    
    # Access mutation reasoning
    print("\n4. Mutation Reasoning:")
    for mutation in insights['mutation_reasoning'][:2]:
        print(f"   Iteration {mutation.get('iteration', 'N/A')}:")
        print(f"   Score Change: {mutation.get('from_score', 0):.4f} → {mutation.get('to_score', 0):.4f}")
        if 'reasoning' in mutation:
            print(f"   Why: {mutation['reasoning'][:80]}...")


def example_3_compare_prompts():
    """
    Example 3: Compare prompts across iterations to see what changed
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Prompts Across Iterations")
    print("="*80)
    
    import json
    from pathlib import Path
    
    insights_dir = Path("./example_insights")
    json_files = list(insights_dir.glob("gepa_insights_*.json"))
    
    if not json_files:
        print("No insights files found. Run example_1_basic_extraction() first.")
        return
    
    with open(json_files[0]) as f:
        insights = json.load(f)
    
    # Compare mutation changes
    print("\nPrompt Evolution:")
    for mutation in insights['mutation_reasoning']:
        if 'prompt_before' in mutation and 'prompt_after' in mutation:
            print(f"\nIteration {mutation['iteration']}:")
            print(f"  Score: {mutation.get('from_score', 0):.4f} → {mutation.get('to_score', 0):.4f}")
            print(f"  Before: {mutation['prompt_before'][:60]}...")
            print(f"  After:  {mutation['prompt_after'][:60]}...")
            print(f"  Change: {mutation.get('change_summary', 'N/A')}")


def example_4_save_and_load_results():
    """
    Example 4: Save result object and analyze later
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Save and Load Results")
    print("="*80)
    
    # Create mock result
    class MockResult:
        def __init__(self):
            self.best_score = 0.8567
            self.best_candidate = {
                'system_prompt': 'Optimized prompt for house prices competition...'
            }
            self.history = []
    
    result = MockResult()
    
    # Save result
    result_file = "./example_result.pkl"
    save_result_to_file(result, result_file)
    print(f"✓ Result saved to {result_file}")
    
    # Load result later
    loaded_result = load_result_from_file(result_file)
    print(f"✓ Result loaded successfully")
    print(f"  Best score: {loaded_result.best_score:.4f}")
    
    # Analyze loaded result
    insights = analyze_gepa_result(
        loaded_result,
        competition_name="house-prices",
        output_dir="./example_insights"
    )
    print(f"✓ Insights extracted from loaded result")


def example_5_inspect_structure():
    """
    Example 5: Inspect result structure to understand available data
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Inspect Result Structure")
    print("="*80)
    
    # Create mock result
    class MockResult:
        def __init__(self):
            self.best_score = 0.8567
            self.best_candidate = {'system_prompt': 'Test prompt'}
            self.history = [{'iteration': 0, 'score': 0.8}]
            self.metadata = {'optimizer': 'GEPA', 'version': '1.0'}
    
    result = MockResult()
    
    # Inspect structure
    inspect_result_structure(result)


def example_6_filter_best_iterations():
    """
    Example 6: Filter and analyze only the best-performing iterations
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Filter Best Iterations")
    print("="*80)
    
    import json
    from pathlib import Path
    
    insights_dir = Path("./example_insights")
    json_files = list(insights_dir.glob("gepa_insights_*.json"))
    
    if not json_files:
        print("No insights files found. Run example_1_basic_extraction() first.")
        return
    
    with open(json_files[0]) as f:
        insights = json.load(f)
    
    # Filter iterations with significant improvements
    print("\nIterations with Significant Improvements:")
    
    performance = insights['performance_history']
    if len(performance) > 1:
        for i in range(1, len(performance)):
            prev_entry = performance[i-1]
            curr_entry = performance[i]
            
            if 'summary' in curr_entry:
                continue  # Skip summary entry
            
            prev_score = prev_entry.get('score', 0)
            curr_score = curr_entry.get('score', 0)
            
            improvement = curr_score - prev_score
            if improvement > 0.01:  # More than 1% improvement
                print(f"  Iteration {i}: {prev_score:.4f} → {curr_score:.4f} (+{improvement:.4f})")
                
                # Find corresponding reflection
                for reflection in insights['reflection_insights']:
                    if reflection.get('iteration') == i:
                        if 'reflection_text' in reflection:
                            print(f"    Insight: {reflection['reflection_text'][:80]}...")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*80)
    print("GEPA INSIGHTS EXTRACTION - EXAMPLES")
    print("="*80)
    
    examples = [
        example_1_basic_extraction,
        example_2_access_specific_insights,
        example_3_compare_prompts,
        example_4_save_and_load_results,
        example_5_inspect_structure,
        example_6_filter_best_iterations
    ]
    
    for example in examples:
        try:
            example()
            print("\n✓ Example completed successfully")
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*80)
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nOutput locations:")
    print("  - ./example_insights/        - Insights files")
    print("  - ./example_result.pkl       - Saved result object")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    # Or run individual examples:
    # example_1_basic_extraction()
    # example_2_access_specific_insights()
    # etc.
