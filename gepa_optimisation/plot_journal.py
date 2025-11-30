"""
Plot journal data from CSV files with dark theme and pink/green dots.
Usage: python plot_journal.py [path_to_journal_history.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_journal(csv_path: str = None):
    """
    Plot journal node scores with pink dots for buggy nodes and green for successful ones.
    
    Args:
        csv_path: Path to journal_history.csv file. If None, uses default location.
    """
    # Default path if not specified
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "prepared" / "journal_logs" / "journal_history.csv"
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Make sure you've run the adapter first to generate the journal data.")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} nodes from {csv_path}")
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Split data by buggy status
    buggy_nodes = df[df['buggy'] == True]
    good_nodes = df[df['buggy'] == False]
    
    # Plot with pink (buggy) and green (successful) dots
    if len(buggy_nodes) > 0:
        ax.scatter(buggy_nodes['node_id'], buggy_nodes['score'], 
                  c='#FF69B4', s=100, alpha=0.7, label='Buggy/Failed', 
                  edgecolors='white', linewidth=0.5)
    
    if len(good_nodes) > 0:
        ax.scatter(good_nodes['node_id'], good_nodes['score'], 
                  c='#00FF7F', s=100, alpha=0.7, label='Successful', 
                  edgecolors='white', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Journal Node Scores by Episode', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add episode separators if multiple episodes
    if 'episode' in df.columns and df['episode'].nunique() > 1:
        episodes = df['episode'].unique()
        for episode in episodes[1:]:  # Skip first episode
            first_node_in_episode = df[df['episode'] == episode]['node_id'].min()
            ax.axvline(x=first_node_in_episode - 0.5, color='yellow', 
                      linestyle='--', alpha=0.5, linewidth=1)
    
    # Tight layout and show
    plt.tight_layout()
    
    # Save plot
    output_path = csv_path.parent / 'journal_plot.png'
    plt.savefig(output_path, dpi=150, facecolor='#1e1e1e', edgecolor='none')
    print(f"Plot saved to {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # Allow custom CSV path from command line
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    plot_journal(csv_path)
