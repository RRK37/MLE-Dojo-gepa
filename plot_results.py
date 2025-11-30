import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Read the CSV data
df = pd.read_csv('journal_nodes.csv')

# Fix episode counting - assign candidate_id based on timestamp groups
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
df = df.sort_values('timestamp').reset_index(drop=True)

# Group by timestamp windows (episodes from same candidate are within ~5 minutes)
time_diff = df['timestamp'].diff().dt.total_seconds()
candidate_breaks = (time_diff > 300) | (time_diff.isna())  # 5 min threshold
df['candidate_id'] = candidate_breaks.cumsum()

# Within each candidate, renumber episodes sequentially
episode_mapping = {}
for (cand, ep), group in df.groupby(['candidate_id', 'episode']):
    if cand not in episode_mapping:
        episode_mapping[cand] = {}
    if ep not in episode_mapping[cand]:
        episode_mapping[cand][ep] = len(episode_mapping[cand])

df['corrected_episode'] = df.apply(
    lambda row: episode_mapping[row['candidate_id']][row['episode']], 
    axis=1
)

# Create a global episode ID
df['global_episode'] = df.groupby(['candidate_id', 'corrected_episode']).ngroup()

print(f"Found {df['candidate_id'].nunique()} candidates")
print(f"Found {df['global_episode'].nunique()} total episodes")
print("\nCandidate breakdown:")
print(df.groupby('candidate_id')['corrected_episode'].nunique())

# ============================================
# Plot 1: Per-Candidate Average Best Score
# ============================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

# For each candidate, get the best score from each episode, then average
candidate_stats = []
for cand_id in sorted(df['candidate_id'].unique()):
    cand_data = df[df['candidate_id'] == cand_id]
    
    # Get best score per episode
    episode_best_scores = []
    for ep in cand_data['corrected_episode'].unique():
        ep_data = cand_data[cand_data['corrected_episode'] == ep]
        best_score = ep_data[ep_data['status'] == 'SUCCESS']['score'].max()
        if pd.notna(best_score):
            episode_best_scores.append(best_score)
    
    if episode_best_scores:
        avg_best = np.mean(episode_best_scores)
        std_best = np.std(episode_best_scores)
        candidate_stats.append({
            'candidate_id': cand_id,
            'avg_best_score': avg_best,
            'std_best_score': std_best,
            'num_episodes': len(episode_best_scores)
        })

stats_df = pd.DataFrame(candidate_stats)

# Plot with error bars
ax1.errorbar(stats_df['candidate_id'], 
             stats_df['avg_best_score'],
             yerr=stats_df['std_best_score'],
             marker='o', 
             markersize=8,
             linewidth=2,
             capsize=5,
             label='Average Best Score')

ax1.set_xlabel('Candidate ID (GEPA Generation)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Best Score', fontsize=12, fontweight='bold')
ax1.set_title('GEPA Optimization Progress: Average Best Score per Candidate', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add value labels
for _, row in stats_df.iterrows():
    ax1.text(row['candidate_id'], 
             row['avg_best_score'] + 0.005, 
             f"{row['avg_best_score']:.4f}",
             ha='center', 
             va='bottom',
             fontsize=8)

plt.tight_layout()
plt.savefig('candidate_progress.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: candidate_progress.png")

# ============================================
# Plot 2: Heatmap - Node Scores by Candidate
# ============================================
fig2, ax2 = plt.subplots(figsize=(14, 10))

# Create pivot table: candidates x nodes, averaging scores across episodes
pivot_data = df[df['status'] == 'SUCCESS'].pivot_table(
    index='candidate_id',
    columns='node_id',
    values='score',
    aggfunc='mean'
)

# Create heatmap
sns.heatmap(pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn',
            vmin=0.78,
            vmax=0.84,
            cbar_kws={'label': 'Average Score'},
            linewidths=0.5,
            ax=ax2)

ax2.set_xlabel('Node ID (Iteration within Episode)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Candidate ID (GEPA Generation)', fontsize=12, fontweight='bold')
ax2.set_title('Score Heatmap: Node Performance Across Candidates\n(Averaged over episodes)', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('candidate_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: candidate_heatmap.png")

# ============================================
# Bonus: Episode-level heatmap with all data
# ============================================
fig3, ax3 = plt.subplots(figsize=(14, 12))

# Create a compound index for global_episode and node_id
episode_pivot = df[df['status'] == 'SUCCESS'].pivot_table(
    index='global_episode',
    columns='node_id',
    values='score',
    aggfunc='first'  # One score per episode-node pair
)

sns.heatmap(episode_pivot, 
            annot=False,  # Too many cells to annotate
            cmap='RdYlGn',
            vmin=0.78,
            vmax=0.84,
            cbar_kws={'label': 'Score'},
            linewidths=0,
            ax=ax3)

ax3.set_xlabel('Node ID (Iteration)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Global Episode ID', fontsize=12, fontweight='bold')
ax3.set_title('Detailed Score Heatmap: All Episodes and Nodes', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('episode_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: episode_heatmap.png")

# ============================================
# Summary Statistics
# ============================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Candidates: {df['candidate_id'].nunique()}")
print(f"Total Episodes: {df['global_episode'].nunique()}")
print(f"Total Nodes: {len(df)}")
print(f"Success Rate: {(df['status'] == 'SUCCESS').mean():.1%}")
print(f"\nOverall Best Score: {df[df['status'] == 'SUCCESS']['score'].max():.4f}")
print(f"Overall Average Score: {df[df['status'] == 'SUCCESS']['score'].mean():.4f}")

print("\nPer-Candidate Statistics:")
print(stats_df.to_string(index=False))

# Find best candidate
best_cand = stats_df.loc[stats_df['avg_best_score'].idxmax()]
print(f"\n🏆 Best Candidate: {int(best_cand['candidate_id'])} "
      f"(avg score: {best_cand['avg_best_score']:.4f} ± {best_cand['std_best_score']:.4f})")

plt.show()
