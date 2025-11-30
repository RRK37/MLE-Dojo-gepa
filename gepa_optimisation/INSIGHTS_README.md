# GEPA Optimization Insights Extraction

This module provides tools to extract and analyze insights from GEPA optimization results, including:
- **Best prompt found** during optimization
- **Reflection insights** - what the LLM learned about what works and what doesn't
- **Mutation reasoning** - why specific prompt changes were made
- **Performance trends** - score progression across iterations

## 📁 Files

- **`extract_insights.py`** - Core extraction class that parses GEPA results
- **`analyze_results.py`** - Standalone script for analyzing saved results
- **`run_opt.py`** - Updated optimization script with automatic insights extraction

## 🚀 Quick Start

### Automatic Extraction (Integrated)

When you run the optimization, insights are automatically extracted:

```python
python gepa_optimisation/run_opt.py
```

After optimization completes, you'll find:
- `gepa_insights/gepa_insights_<timestamp>.json` - Full insights in JSON format
- `gepa_insights/gepa_report_<timestamp>.txt` - Human-readable report
- `best_prompt.txt` - Quick access to the best prompt

### Manual Analysis

To analyze a saved result object:

```python
from gepa_optimisation.analyze_results import analyze_gepa_result

# After running optimize()
result = optimize(...)

# Extract insights
insights = analyze_gepa_result(
    result, 
    competition_name="titanic",
    output_dir="./gepa_insights"
)
```

### Command-Line Analysis

If you saved a result object to a file:

```bash
# Save result during optimization
python -c "import pickle; pickle.dump(result, open('result.pkl', 'wb'))"

# Analyze later
python gepa_optimisation/analyze_results.py --result-file result.pkl --competition titanic

# Inspect result structure
python gepa_optimisation/analyze_results.py --result-file result.pkl --inspect
```

## 📊 What Gets Extracted

### 1. Best Prompt
The highest-scoring system prompt found during optimization.

```json
{
  "best_prompt": "You are a Kaggle Grandmaster...",
  "best_score": 0.8342
}
```

### 2. Reflection Insights
GEPA's LLM analyzes what worked and what didn't across iterations:

```json
{
  "reflection_insights": [
    {
      "iteration": 3,
      "score": 0.8234,
      "reflection_text": "Adding cross-validation improved robustness...",
      "trajectory_summary": "12 total steps, no errors"
    }
  ]
}
```

### 3. Mutation Reasoning
Why prompts were changed between iterations:

```json
{
  "mutation_reasoning": [
    {
      "iteration": 2,
      "from_score": 0.8012,
      "to_score": 0.8234,
      "reasoning": "Agent struggled with feature selection, added explicit guidance",
      "change_summary": "Length increased by 147 chars, added: cross-validation, feature engineering"
    }
  ]
}
```

### 4. Performance History
Score progression and statistics:

```json
{
  "performance_history": [
    {"iteration": 0, "score": 0.7856},
    {"iteration": 1, "score": 0.8012},
    {"iteration": 2, "score": 0.8234},
    {
      "summary": {
        "total_iterations": 3,
        "initial_score": 0.7856,
        "final_score": 0.8234,
        "improvement": 0.0378,
        "improvement_pct": 4.81,
        "best_score": 0.8234,
        "avg_score": 0.8034
      }
    }
  ]
}
```

## 📝 Output Formats

### JSON Format
Complete structured data for programmatic access:
```
gepa_insights/gepa_insights_20241130_143022.json
```

### Human-Readable Report
Formatted text report for easy reading:
```
gepa_insights/gepa_report_20241130_143022.txt
```

Example:
```
================================================================================
GEPA OPTIMIZATION INSIGHTS REPORT
================================================================================

Competition: titanic
Timestamp: 20241130_143022

--------------------------------------------------------------------------------
BEST RESULTS
--------------------------------------------------------------------------------
Best Score: 0.8342

Best Prompt:
You are a Kaggle Grandmaster with expertise in tabular data...

--------------------------------------------------------------------------------
REFLECTION INSIGHTS
--------------------------------------------------------------------------------

Iteration 1:
  Score: 0.8012
  Reflection: Initial approach worked but lacked feature engineering...
  
Iteration 2:
  Score: 0.8234
  Reflection: Cross-validation significantly improved robustness...
```

## 🔍 Inspecting Result Structure

To understand what data is available in your GEPA results:

```python
from gepa_optimisation.analyze_results import inspect_result_structure

inspect_result_structure(result)
```

This prints:
- Result object type and attributes
- History structure and length
- Available metadata
- Sample entries

## 🎯 Use Cases

### 1. Understanding Optimization Progress
Track how the optimizer learned and improved prompts over time.

### 2. Debugging Failed Optimizations
Inspect what went wrong by examining reflection insights and trajectories.

### 3. Reproducing Best Results
Extract the exact prompt that achieved the best score.

### 4. Learning from GEPA
Understand what prompt patterns work best for your domain.

### 5. Building Better Initial Prompts
Use insights from previous optimizations to craft better starting prompts.

## 🛠️ Advanced Usage

### Custom Insight Extraction

```python
from gepa_optimisation.extract_insights import GEPAInsightsExtractor

# Create extractor with custom output directory
extractor = GEPAInsightsExtractor(output_dir="./my_insights")

# Extract insights
insights = extractor.extract_and_save(result, competition_name="titanic")

# Access specific insights
best_prompt = insights['best_prompt']
reflections = insights['reflection_insights']
mutations = insights['mutation_reasoning']
```

### Saving Results for Later Analysis

```python
from gepa_optimisation.analyze_results import save_result_to_file, load_result_from_file

# During optimization
result = optimize(...)
save_result_to_file(result, "optimization_results.pkl")

# Later
result = load_result_from_file("optimization_results.pkl")
insights = analyze_gepa_result(result, competition_name="titanic")
```

### Comparing Multiple Optimization Runs

```python
import json
from pathlib import Path

# Load multiple insight files
insights_dir = Path("./gepa_insights")
all_insights = []

for insight_file in insights_dir.glob("gepa_insights_*.json"):
    with open(insight_file) as f:
        all_insights.append(json.load(f))

# Compare best scores
for insights in all_insights:
    print(f"{insights['metadata']['timestamp']}: {insights['best_score']:.4f}")
```

## 📈 Visualization

The insights data can be visualized using the journal plotting tools:

```python
from gepa_optimisation.plot_journal import plot_journal_history

# Plot performance trends
plot_journal_history("./output/journal_logs/journal_history.csv")
```

## ⚙️ Configuration

The insights extractor can be customized:

```python
extractor = GEPAInsightsExtractor(
    output_dir="./custom_insights"  # Where to save reports
)

# Customize what gets extracted by modifying the extractor methods
```

## 🐛 Troubleshooting

### No Reflection Data Found
If `reflection_insights` is empty, GEPA might not be storing reflection data in `result.history`. Check:
- GEPA version and documentation
- Whether `capture_traces=True` in adapter
- Result structure with `inspect_result_structure()`

### Missing History
If `result.history` is None or empty:
- Check GEPA configuration
- Ensure optimization completed successfully
- Verify GEPA is configured to store history

### Extraction Errors
If extraction fails:
- Use `inspect_result_structure()` to check result format
- Review error messages in console output
- Check that result object is from a completed optimization

## 📚 Related Files

- `adapter.py` - GEPA adapter with trajectory capture
- `plot_journal.py` - Visualization tools for journal data
- `run_opt.py` - Main optimization script

## 🤝 Contributing

To extend the insights extractor:

1. Add new extraction methods to `GEPAInsightsExtractor`
2. Update the report formatting in `_save_human_readable_report()`
3. Add new fields to the insights dictionary structure

## 📄 License

Same as parent project.
