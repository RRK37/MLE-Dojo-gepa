# Pattern Analysis System for AIDE Agent

## Overview

The pattern analysis system automatically learns from the execution of ML solutions by analyzing patterns that correlate with success or failure. It extracts features from code, clusters similar solutions, and generates actionable insights that are injected into future prompts to guide the agent toward better solutions.

## Architecture

### Components

1. **CodeFeatureExtractor**
   - Detects model types (xgboost, lightgbm, random_forest, neural_network, etc.)
   - Identifies preprocessing techniques (scaling, encoding, imputation, feature engineering)
   - Recognizes validation strategies (cross-validation, train_test_split, stratification)
   - Detects hyperparameter optimization methods (grid_search, random_search, optuna, bayesian)

2. **SolutionAnalyzer**
   - Analyzes individual solutions by extracting code features
   - Associates features with execution metrics (score, execution time)
   - Converts feature dictionaries to numerical vectors for clustering

3. **PatternDiscovery**
   - Clusters solutions using KMeans to group similar approaches
   - Identifies distinct solution strategies (e.g., "tree-based models with grid search")
   - Enables comparison of different solution families

4. **CorrelationAnalyzer**
   - Analyzes clusters to identify performance patterns:
     - **High-performing patterns**: Consistently good results
     - **Low-performing patterns**: Consistently poor results
     - **Risky patterns**: High variance (sometimes great, sometimes poor)
     - **Stable patterns**: Low variance (consistent results)
   - Provides statistical evidence for what works and what doesn't

5. **InsightGenerator**
   - Converts statistical analysis into human-readable insights
   - Generates actionable guidance for future solutions
   - Examples:
     - "Avoid: Solutions using model_type=random_forest and preprocessing=imputation tend to underperform"
     - "Recommended: Solutions with model_type=xgboost and validation=cross_validation show 23% better average scores"

6. **TemporalPatternAnalyzer**
   - Detects stagnation (scores not improving)
   - Measures solution diversity over time
   - Alerts when the agent is stuck in local optima

## Integration with AIDE Agent

### Initialization
The pattern analysis components are initialized when the Agent is created:
```python
self.solution_analyzer = SolutionAnalyzer()
self.pattern_discovery = PatternDiscovery()
self.correlation_analyzer = CorrelationAnalyzer()
self.insight_generator = InsightGenerator()
self.temporal_analyzer = TemporalPatternAnalyzer()

self.learned_insights = ""
self.learned_guidance = []
self.last_analysis_step = -1
```

### Analysis Trigger
Pattern analysis runs every 3 steps via `_update_learned_insights()`:
- Collects all successful (non-buggy) solutions from the journal
- Extracts features from each solution's code
- Clusters solutions to find patterns
- Analyzes correlation between patterns and performance
- Generates insights and guidance
- Performs temporal analysis to detect stagnation

### Prompt Injection
Learned insights are injected into prompts for both `_draft()` and `_improve()`:

```python
if self.learned_insights:
    prompt["Learned Patterns"] = {
        "Analysis of previous solutions": self.learned_insights,
        "Actionable guidance": self.learned_guidance,
    }

if self.learned_guidance:
    prompt["Instructions"]["Apply learned patterns"] = self.learned_guidance
```

This ensures the LLM has access to learned patterns when generating new solutions.

## Workflow

1. **Solution Execution**: Agent generates and executes solutions
2. **Feature Extraction**: After each execution, code features are extracted
3. **Periodic Analysis**: Every 3 steps, pattern analysis runs:
   - Solutions are clustered by similarity
   - Clusters are analyzed for performance patterns
   - Insights are generated
4. **Prompt Enhancement**: Next solutions receive learned insights in their prompts
5. **Continuous Learning**: The cycle repeats, accumulating knowledge over time

## Benefits

### Automatic Learning
- No manual intervention required
- Learns what works for specific competition types
- Adapts to dataset characteristics

### Exploration vs Exploitation
- Identifies "risky but potentially high-reward" patterns
- Warns against consistently underperforming approaches
- Balances trying new things with leveraging proven strategies

### Stagnation Detection
- Detects when the agent is stuck
- Triggers exploration when diversity decreases
- Prevents wasted iterations on similar approaches

### Knowledge Transfer
- Insights can be analyzed across competitions
- Patterns from one competition may inform another
- Builds a library of ML best practices

## Example Insights

**High-Performing Pattern:**
```
Cluster 1 (3 solutions):
- Average score: 0.87
- Common features: xgboost, cross_validation, feature_engineering
- Guidance: "Continue using XGBoost with cross-validation and feature engineering"
```

**Low-Performing Pattern:**
```
Cluster 2 (2 solutions):
- Average score: 0.42
- Common features: random_forest, no validation, basic preprocessing
- Guidance: "Avoid random_forest without proper validation - scores 52% lower than best approaches"
```

**Temporal Alert:**
```
Stagnation detected: Last 5 solutions show no improvement
Diversity decreasing: Solutions becoming too similar
Recommendation: Try a fundamentally different approach (e.g., neural network instead of tree-based)
```

## Configuration

### Analysis Frequency
Currently set to run every 3 steps. Can be adjusted in `_update_learned_insights()`:
```python
if current_step - self.last_analysis_step < 3 or current_step < 3:
    return
```

### Clustering Parameters
Number of clusters adapts to number of solutions:
```python
clusters = self.pattern_discovery.cluster_solutions(
    solutions,
    n_clusters=min(3, len(solutions))
)
```

### Feature Weights
All features currently have equal weight. Could be enhanced to prioritize certain features based on importance.

## Future Enhancements

1. **Meta-Learning**: Store patterns across competitions to build a knowledge base
2. **Feature Importance**: Weight features by their correlation with success
3. **Causal Analysis**: Identify causal relationships, not just correlations
4. **Ensemble Recommendations**: Suggest which solutions to ensemble based on diversity
5. **Active Learning**: Propose experiments to test hypotheses about what works
6. **Transfer Learning**: Apply insights from similar competitions

## Dependencies

- `scikit-learn`: KMeans clustering, TF-IDF vectorization
- `numpy`: Numerical operations
- `typing`: Type hints
- Standard library: `re`, `collections`

## Testing

To verify the pattern analysis system is working:
1. Run the agent on a competition with multiple iterations
2. Check logs for "Running pattern analysis at step X" messages
3. Examine generated prompts for "Learned Patterns" section
4. Verify insights become more specific as more solutions are analyzed

## Troubleshooting

If pattern analysis fails:
- Check logs for warning messages
- Ensure at least 3 successful (non-buggy) solutions exist
- Verify scikit-learn is installed
- Pattern analysis failures won't crash the agent (wrapped in try-except)
