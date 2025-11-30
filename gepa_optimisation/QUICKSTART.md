# GEPA Insights Extraction - Quick Reference

## 🎯 What You Get

After running GEPA optimization, you automatically get:

1. **Best Prompt Found** - The highest-scoring system prompt
2. **Reflection Insights** - What the LLM learned (e.g., "Cross-validation improved scores by 0.02")
3. **Mutation Reasoning** - Why prompts were changed (e.g., "Agent struggled with feature selection")
4. **Performance Trends** - Score progression and statistics

## 🚀 Quick Start

### Run Optimization (Automatic Extraction)
```bash
python gepa_optimisation/run_opt.py
```

**Output:**
- `gepa_insights/gepa_insights_<timestamp>.json` - Full data
- `gepa_insights/gepa_report_<timestamp>.txt` - Human-readable
- `best_prompt.txt` - Quick access to best prompt

### View Results
```bash
# Open the human-readable report
cat gepa_insights/gepa_report_*.txt

# Or view JSON for programmatic access
cat gepa_insights/gepa_insights_*.json
```

## 📋 Example Output

```
================================================================================
GEPA OPTIMIZATION INSIGHTS
================================================================================

📊 Competition: titanic
🏆 Best Score: 0.8342
🔁 Total Iterations: 5

--------------------------------------------------------------------------------
✨ BEST PROMPT FOUND:
--------------------------------------------------------------------------------
You are a Kaggle Grandmaster. Focus on feature engineering and robust 
validation. Always create submission.csv with predictions.

--------------------------------------------------------------------------------
🔍 KEY REFLECTION INSIGHTS:
--------------------------------------------------------------------------------

Iteration 1 (Score: 0.8012):
  Initial baseline established. Agent needs better feature engineering guidance.

Iteration 2 (Score: 0.8234):
  Cross-validation implementation improved robustness by 0.0222.

Iteration 3 (Score: 0.8342):
  Feature engineering emphasis led to significant score improvement.

--------------------------------------------------------------------------------
🔄 MUTATION REASONING:
--------------------------------------------------------------------------------

Iteration 2:
  Score: 0.8012 → 0.8234
  Why: Added explicit cross-validation instructions
  Changes: Length increased by 47 chars, added: cross-validation

Iteration 3:
  Score: 0.8234 → 0.8342
  Why: Emphasized feature engineering and submission format
  Changes: Length increased by 123 chars, added: feature engineering, submission
```

## 🔧 Programmatic Access

```python
import json

# Load insights
with open('gepa_insights/gepa_insights_<timestamp>.json') as f:
    insights = json.load(f)

# Access data
best_prompt = insights['best_prompt']
best_score = insights['best_score']
reflections = insights['reflection_insights']
mutations = insights['mutation_reasoning']

# Example: Print all reflection insights
for reflection in reflections:
    print(f"Iteration {reflection['iteration']}: {reflection['reflection_text']}")
```

## 📁 Files Created

| File | Description | Format |
|------|-------------|--------|
| `gepa_insights_*.json` | Complete insights data | JSON |
| `gepa_report_*.txt` | Human-readable report | Text |
| `best_prompt.txt` | Best prompt only | Text |
| `journal_history.json` | Agent node history | JSON |
| `journal_history.csv` | Agent node data | CSV |

## 💡 Use Cases

- **Understand what worked** - See which prompt changes improved scores
- **Debug failed runs** - Identify what went wrong
- **Learn patterns** - Build intuition about effective prompts
- **Reproduce results** - Get exact prompts for best scores
- **Compare runs** - Track improvements across optimization sessions

## 📚 More Information

- **Full documentation**: `gepa_optimisation/INSIGHTS_README.md`
- **Examples**: `gepa_optimisation/example_usage.py`
- **Standalone analysis**: `gepa_optimisation/analyze_results.py`

## ⚡ Advanced Usage

### Save Result for Later Analysis
```python
from gepa_optimisation.analyze_results import save_result_to_file

result = optimize(...)
save_result_to_file(result, "my_result.pkl")
```

### Analyze Saved Result
```bash
python gepa_optimisation/analyze_results.py --result-file my_result.pkl --competition titanic
```

### Inspect Result Structure
```python
from gepa_optimisation.analyze_results import inspect_result_structure

inspect_result_structure(result)
```

### Run Examples
```bash
python gepa_optimisation/example_usage.py
```

## 🎓 Next Steps

1. Run optimization: `python gepa_optimisation/run_opt.py`
2. Check outputs in `gepa_insights/` directory
3. Review the human-readable report
4. Use insights to improve future optimizations
5. Explore programmatic access for automation
