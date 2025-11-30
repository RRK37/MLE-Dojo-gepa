# GEPA Insights Extraction - Architecture & Flow

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GEPA Optimization                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ run_opt.py                                               │  │
│  │                                                          │  │
│  │  1. Setup configuration                                 │  │
│  │  2. Create agent factory                                │  │
│  │  3. Initialize MLEDojoGEPAAdapter                       │  │
│  │  4. Run optimize()  ─────────────────┐                  │  │
│  │                                       │                  │  │
│  │  5. Get result object  ◄─────────────┘                  │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  6. Extract insights  ──────────────►┌──────────────┐   │  │
│  │                                      │  Extractor   │   │  │
│  │  7. Save & display   ◄───────────────└──────────────┘   │  │
│  │         │                                                │  │
│  └─────────┼────────────────────────────────────────────────┘  │
└────────────┼───────────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  Output Files          │
    │                        │
    │  • gepa_insights_*.json │
    │  • gepa_report_*.txt    │
    │  • best_prompt.txt      │
    └────────────────────────┘
```

## 🔄 Data Flow

```
GEPA optimize()
       │
       ├─► result.best_candidate ──┐
       │                            │
       ├─► result.best_score ───────┼──► GEPAInsightsExtractor
       │                            │           │
       └─► result.history ──────────┘           │
              │                                  │
              ├─► iteration 0                    ▼
              │     ├─ score                ┌─────────┐
              │     ├─ candidate            │ Analyze │
              │     ├─ reflection           │ Process │
              │     └─ mutation_reason      └─────────┘
              │                                  │
              ├─► iteration 1                    │
              │     ├─ score                     │
              │     ├─ candidate                 │
              │     ├─ reflection                │
              │     └─ mutation_reason           │
              │                                  │
              └─► iteration N                    │
                    └─ ...                       │
                                                 │
                ┌────────────────────────────────┘
                │
                ▼
        ┌───────────────────┐
        │  Insights Object  │
        │                   │
        │  • best_prompt    │
        │  • best_score     │
        │  • reflections    │
        │  • mutations      │
        │  • performance    │
        │  • summary        │
        └───────────────────┘
                │
                ├──► Save JSON
                ├──► Save Text Report
                └──► Print Console
```

## 🗂️ File Structure

```
mle-RL-gepa/
└── MLE-Dojo-gepa/
    ├── gepa_optimisation/
    │   ├── run_opt.py                    # Main optimization script
    │   ├── adapter.py                    # GEPA adapter
    │   │
    │   ├── extract_insights.py           # 🆕 Core extraction class
    │   ├── analyze_results.py            # 🆕 Standalone analysis
    │   ├── example_usage.py              # 🆕 Usage examples
    │   │
    │   ├── INSIGHTS_README.md            # 🆕 Full documentation
    │   ├── QUICKSTART.md                 # 🆕 Quick reference
    │   └── IMPLEMENTATION_SUMMARY.md     # 🆕 Implementation summary
    │
    ├── gepa_insights/                    # 🆕 Output directory
    │   ├── gepa_insights_*.json          # Structured insights
    │   ├── gepa_report_*.txt             # Human-readable
    │   └── live_plot.png                 # Visualization
    │
    ├── output/
    │   └── journal_logs/                 # Agent execution logs
    │       ├── journal_history.json
    │       └── journal_history.csv
    │
    └── best_prompt.txt                   # 🆕 Quick-access best prompt
```

## 🔍 Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    GEPAInsightsExtractor                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  extract_and_save(result, competition_name)                     │
│         │                                                        │
│         ├─► _extract_best_prompt()                              │
│         │        └─► result.best_candidate['system_prompt']     │
│         │                                                        │
│         ├─► _extract_best_score()                               │
│         │        └─► result.best_score                          │
│         │                                                        │
│         ├─► _extract_reflection_insights()                      │
│         │        └─► Iterate result.history                     │
│         │              ├─ entry['reflection']                   │
│         │              ├─ entry['analysis']                     │
│         │              ├─ entry['feedback']                     │
│         │              └─ entry['trajectories']                 │
│         │                                                        │
│         ├─► _extract_mutation_reasoning()                       │
│         │        └─► Compare consecutive history entries        │
│         │              ├─ Prompt before/after                   │
│         │              ├─ Score change                          │
│         │              ├─ entry['mutation_reason']              │
│         │              └─ entry['explanation']                  │
│         │                                                        │
│         ├─► _extract_performance_history()                      │
│         │        └─► Iterate result.history                     │
│         │              ├─ Score per iteration                   │
│         │              └─ Calculate statistics                  │
│         │                                                        │
│         ├─► _create_summary()                                   │
│         │        └─► Analyze trends                             │
│         │              ├─ Total iterations                      │
│         │              ├─ Biggest improvements                  │
│         │              └─ Convergence analysis                  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Insights   │                                               │
│  │   Object     │                                               │
│  └──────────────┘                                               │
│         │                                                        │
│         ├─► _save_human_readable_report()                       │
│         │        └─► gepa_report_*.txt                          │
│         │                                                        │
│         ├─► Save JSON                                           │
│         │        └─► gepa_insights_*.json                       │
│         │                                                        │
│         └─► _print_insights()                                   │
│                  └─► Console output                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Usage Patterns

### Pattern 1: Automatic (Default)
```python
# In run_opt.py
result = optimize(...)

# Automatic extraction happens here
extractor = GEPAInsightsExtractor(output_dir="./gepa_insights")
insights = extractor.extract_and_save(result, competition_name="titanic")
```

### Pattern 2: Manual Analysis
```python
from gepa_optimisation.analyze_results import analyze_gepa_result

# Load or get result
result = ...

# Analyze anytime
insights = analyze_gepa_result(result, competition_name="titanic")
```

### Pattern 3: Save & Load
```python
from gepa_optimisation.analyze_results import (
    save_result_to_file,
    load_result_from_file
)

# After optimization
save_result_to_file(result, "my_result.pkl")

# Later...
result = load_result_from_file("my_result.pkl")
insights = analyze_gepa_result(result)
```

### Pattern 4: Command-Line
```bash
# Save during optimization
python run_opt.py  # result auto-saved

# Analyze from command line
python analyze_results.py --result-file result.pkl --competition titanic
```

## 📈 Data Transformation

```
Raw GEPA Result
       │
       ▼
┌─────────────────┐
│ result.history  │
├─────────────────┤
│ [               │
│   {             │
│     iteration: 0│
│     score: 0.78 │
│     candidate:  │──────┐
│       {...}     │      │
│     reflection: │      │    Extract & Transform
│       "..."     │      │            │
│     mutation:   │      │            ▼
│       "..."     │      │    ┌──────────────────┐
│   },            │      │    │ Structured       │
│   {             │      │    │ Insights         │
│     iteration: 1│──────┼───►│                  │
│     score: 0.82 │      │    │ • Organized      │
│     ...         │      │    │ • Summarized     │
│   },            │      │    │ • Enriched       │
│   ...           │      │    │ • Formatted      │
│ ]               │      │    └──────────────────┘
└─────────────────┘      │            │
                         │            │
                         │            ▼
                         │    ┌──────────────────┐
                         │    │ Multiple Outputs │
                         │    ├──────────────────┤
                         └───►│ • JSON (machine) │
                              │ • Text (human)   │
                              │ • Console (dev)  │
                              └──────────────────┘
```

## 🔗 Integration Points

```
┌────────────────────────────────────────────────────────────────┐
│                        Main Script                              │
│                       (run_opt.py)                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Import                                                         │
│  ┌────────────────────────────────────────┐                    │
│  │ from gepa_optimisation.extract_insights│                    │
│  │      import GEPAInsightsExtractor      │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
│  Setup                                                          │
│  ┌────────────────────────────────────────┐                    │
│  │ extractor = GEPAInsightsExtractor(     │                    │
│  │     output_dir="./gepa_insights"       │                    │
│  │ )                                       │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
│  Execute                                                        │
│  ┌────────────────────────────────────────┐                    │
│  │ insights = extractor.extract_and_save( │                    │
│  │     result,                             │                    │
│  │     competition_name="titanic"          │                    │
│  │ )                                       │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 🎨 Output Format Examples

### JSON Output
```json
{
  "metadata": {
    "competition": "titanic",
    "timestamp": "20241130_143022"
  },
  "best_prompt": "You are a Kaggle Grandmaster...",
  "best_score": 0.8342,
  "reflection_insights": [...],
  "mutation_reasoning": [...],
  "performance_history": [...]
}
```

### Text Report
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
You are a Kaggle Grandmaster...
```

### Console Output
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
You are a Kaggle Grandmaster...
```

## ✅ Testing & Validation

```
┌─────────────────────────────────────────────┐
│           example_usage.py                   │
├─────────────────────────────────────────────┤
│                                              │
│  Example 1: Basic extraction                │
│  Example 2: Access specific insights        │
│  Example 3: Compare prompts                 │
│  Example 4: Save and load results           │
│  Example 5: Inspect structure               │
│  Example 6: Filter best iterations          │
│                                              │
└─────────────────────────────────────────────┘
```

Run all examples:
```bash
python gepa_optimisation/example_usage.py
```

## 🎓 Learning Path

1. **Start here**: `QUICKSTART.md`
2. **Try examples**: `python example_usage.py`
3. **Run optimization**: `python run_opt.py`
4. **Review outputs**: Check `gepa_insights/`
5. **Deep dive**: Read `INSIGHTS_README.md`
6. **Customize**: Modify `extract_insights.py`

## 🚀 Next Steps

1. Run the optimization
2. Review generated insights
3. Use insights to improve prompts
4. Compare across multiple runs
5. Automate insight analysis
