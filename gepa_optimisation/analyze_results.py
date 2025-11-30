"""
Standalone script to analyze GEPA optimization results.

This script can be used to analyze saved GEPA results or result objects
to extract insights, reflections, and reasoning.

Usage:
    # From a saved result object
    python analyze_results.py --result-file results.pkl
    
    # From within Python
    from gepa_optimisation.analyze_results import analyze_gepa_result
    insights = analyze_gepa_result(result_object, competition_name="titanic")
"""

import argparse
import pickle
import json
from pathlib import Path
from gepa_optimisation.extract_insights import GEPAInsightsExtractor


def analyze_gepa_result(result, competition_name: str = "unknown", output_dir: str = "./gepa_insights"):
    """
    Analyze a GEPA optimization result object.
    
    Args:
        result: The result object returned by gepa.optimize()
        competition_name: Name of the competition
        output_dir: Directory to save insights
    
    Returns:
        Dictionary containing extracted insights
    """
    extractor = GEPAInsightsExtractor(output_dir=output_dir)
    insights = extractor.extract_and_save(result, competition_name=competition_name)
    return insights


def inspect_result_structure(result):
    """
    Inspect the structure of a GEPA result object to understand its format.
    
    This is useful for debugging and understanding what data is available.
    """
    print("\n" + "="*80)
    print("GEPA RESULT STRUCTURE INSPECTION")
    print("="*80)
    
    print("\n1. Result Type:", type(result))
    
    print("\n2. Result Attributes:")
    if hasattr(result, '__dict__'):
        for attr in dir(result):
            if not attr.startswith('_'):
                try:
                    value = getattr(result, attr)
                    if not callable(value):
                        print(f"   - {attr}: {type(value).__name__}")
                except Exception as e:
                    print(f"   - {attr}: <error accessing: {e}>")
    
    print("\n3. Best Candidate:")
    if hasattr(result, 'best_candidate'):
        print(f"   Type: {type(result.best_candidate)}")
        if isinstance(result.best_candidate, dict):
            print(f"   Keys: {list(result.best_candidate.keys())}")
            if 'system_prompt' in result.best_candidate:
                print(f"   Prompt Preview: {result.best_candidate['system_prompt'][:100]}...")
    
    print("\n4. Best Score:")
    if hasattr(result, 'best_score'):
        print(f"   {result.best_score}")
    
    print("\n5. History:")
    if hasattr(result, 'history'):
        print(f"   Length: {len(result.history) if result.history else 0}")
        if result.history and len(result.history) > 0:
            print(f"   First Entry Type: {type(result.history[0])}")
            if isinstance(result.history[0], dict):
                print(f"   First Entry Keys: {list(result.history[0].keys())}")
            elif hasattr(result.history[0], '__dict__'):
                attrs = [a for a in dir(result.history[0]) if not a.startswith('_')]
                print(f"   First Entry Attributes: {attrs[:10]}")
    
    print("\n6. Other Notable Attributes:")
    notable_attrs = ['state', 'optimizer', 'traces', 'metadata', 'config']
    for attr in notable_attrs:
        if hasattr(result, attr):
            value = getattr(result, attr)
            print(f"   - {attr}: {type(value).__name__}")
    
    print("\n" + "="*80)


def save_result_to_file(result, filepath: str = "gepa_result.pkl"):
    """
    Save a GEPA result object to a file for later analysis.
    
    Args:
        result: The result object to save
        filepath: Path to save the pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Result saved to {filepath}")


def load_result_from_file(filepath: str):
    """
    Load a GEPA result object from a saved file.
    
    Args:
        filepath: Path to the saved pickle file
    
    Returns:
        The loaded result object
    """
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    print(f"✓ Result loaded from {filepath}")
    return result


def main():
    """Command-line interface for analyzing GEPA results."""
    parser = argparse.ArgumentParser(
        description="Analyze GEPA optimization results and extract insights"
    )
    parser.add_argument(
        '--result-file',
        type=str,
        help="Path to saved GEPA result pickle file"
    )
    parser.add_argument(
        '--competition',
        type=str,
        default='unknown',
        help="Competition name"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gepa_insights',
        help="Directory to save insights"
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help="Inspect the result structure instead of extracting insights"
    )
    
    args = parser.parse_args()
    
    if not args.result_file:
        print("Error: --result-file is required")
        print("\nExample usage:")
        print("  python analyze_results.py --result-file gepa_result.pkl --competition titanic")
        return
    
    # Load result
    result = load_result_from_file(args.result_file)
    
    if args.inspect:
        # Just inspect the structure
        inspect_result_structure(result)
    else:
        # Extract full insights
        insights = analyze_gepa_result(
            result,
            competition_name=args.competition,
            output_dir=args.output_dir
        )
        
        print(f"\n✓ Analysis complete! Check {args.output_dir} for detailed reports.")


if __name__ == "__main__":
    main()
