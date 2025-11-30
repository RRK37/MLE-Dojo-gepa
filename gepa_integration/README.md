# GEPA Integration for MLE-Dojo AIDE Agent

This directory contains the implementation of GEPA (Genetic-Pareto) optimization framework integration with MLE-Dojo's AIDE agent for iterative prompt improvement.

## Overview

GEPA optimizes AIDE agent prompts by:
1. Running multiple AIDE agents with different prompt variations on Kaggle competitions
2. Collecting execution traces and performance metrics
3. Building reflective datasets from failures and low-scoring runs
4. Using an LLM to propose improved prompts based on the reflective data
5. Iter atively refining prompts to achieve better competition performance

## Directory Structure

```
gepa_integration/
├── __init__.py                 # Package initialization
├── mledojo_gepa_adapter.py     # GEPA adapter implementation
├── prompt_utils.py             # Prompt management utilities
├── gepa_logger.py              # Logging and visualization
├── config_gepa.yaml            # Configuration file
├── run_gepa_optimization.py    # Main orchestration script
└── README.md                   # This file
```

## Quick Start

### 1. Install GEPA

```bash
pip install gepa
```

### 2. Configure Optimization

Edit `config_gepa.yaml` to set:
- Competitions to use for training and validation
- GEPA hyperparameters (iterations, LLMs)
- AIDE agent settings (steps, timeout)

### 3. Run Optimization

```bash
# Using config file
python gepa_integration/run_gepa_optimization.py --config gepa_integration/config_gepa.yaml

# Or with command-line overrides
python gepa_integration/run_gepa_optimization.py \
    --competition titanic \
    --max-iterations 5 \
    --output-dir ./my_gepa_run
```

## Components

### MLEDojoGEPAAdapter

The core adapter that bridges GEPA with MLE-Dojo:

- **CompetitionConfig**: Defines a single competition run configuration
- **ExecutionTrajectory**: Captures execution traces from AIDE runs
- **evaluate()**: Runs AIDE agents and collects results
- **make_reflective_dataset()**: Builds training data for prompt improvement

### Prompt Management

`prompt_utils.py` provides functions for:
- `extract_default_prompts()`: Get baseline AIDE prompts
- `inject_prompts()`: Apply GEPA-optimized prompts to AIDE
- `validate_prompts()`: Ensure prompt configurations are valid
- `serialize_prompts()` / `deserialize_prompts()`: Save/load prompts

### Logging and Visualization

`gepa_logger.py` tracks:
- Individual AIDE run metrics
- GEPA iteration progress
- Optimization trajectory plots
- Summary reports with best prompts

## Configuration

Key settings in `config_gepa.yaml`:

```yaml
gepa:
  max_iterations: 10              # GEPA iterations
  reflection_lm: "openai/gpt-4o"  # LLM for prompt improvement
  task_lm: "openai/gpt-4o-mini"   # LLM used by AIDE
  train_competitions: [...]        # Training set
  val_competitions: [...]          # Validation set

aide:
  max_steps: 5                    # AIDE steps per run (reduced for speed)
  execution_timeout: 3600         # Timeout per run
```

## Integration with AIDE Agent

The integration modifies AIDE agent to support custom prompt injection:

1. **agent.py**: Added `custom_prompts` parameter to `__init__()`
2. **_draft(), _improve(), _debug()**: Use custom prompts when provided
3. **buildup.py**: Pass custom prompts through setup function
4. **journal.py**: Added GEPA export methods

These changes are **backward compatible** - AIDE works normally when custom_prompts=None.

## Output Structure

```
gepa_outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── logs/
    │   ├── gepa_run_*.log
    │   ├── gepa_run_*_runs.jsonl
    │   ├── gepa_run_*_iterations.jsonl
    │   └── gepa_run_*_progress.png
    ├── seed_prompts.json
    ├── best_prompts.json
    └── gepa_run_*_summary.md
```

## Example: Optimizing Prompts

```python
from gepa_integration.mledojo_gepa_adapter import MLEDojoGEPAAdapter, CompetitionConfig
from gepa_integration.prompt_utils import extract_default_prompts
import gepa

# Setup
base_config = {...}  # Load from config.yaml
adapter = MLEDojoGEPAAdapter(base_config)

# Define competitions
trainset = [
    CompetitionConfig(name="titanic", data_dir="./data/prepared", max_steps=5),
    CompetitionConfig(name="spaceship-titanic", data_dir="./data/prepared", max_steps=5),
]

valset = [
    CompetitionConfig(name="house-prices", data_dir="./data/prepared", max_steps=5),
]

# Get seed prompts
seed_candidate = extract_default_prompts()

# Run optimization
result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o",
    max_metric_calls=10,
    # Note: Full integration requires implementing evaluation wrapper
)

print("Best prompts:", result.best_candidate)
```

## Cost Estimates

**Warning**: GEPA optimization can be expensive!

Example cost for 10 iterations:
- 10 iterations × 2 train competitions × 5 AIDE steps = 100 AIDE agent calls
- Each AIDE call: ~50-100 LLM tokens (gpt-4o-mini)
- GEPA reflections: ~10 calls to gpt-4o for prompt improvement
- **Total estimated cost**: $10-50 depending on LLM usage

## Troubleshooting

### GEPA Not Found
```
pip install gepa
```

### Competition Data Missing
Ensure competition data is downloaded to `data/prepared/<competition_name>/data/`

### Adapter Errors
Check that:
- Base config.yaml is valid
- Environment variables (API keys) are set
- Competition names match directory names

## Advanced Usage

### Custom Reflective Dataset

Modify `MLEDojoGEPAAdapter.make_reflective_dataset()` to customize what feedback is provided to GEPA's reflection LLM.

### Multi-Component Optimization

To optimize multiple prompt components simultaneously, extend the adapter to handle more components beyond the three introduction prompts.

### Custom Evaluation Metrics

The adapter uses position_score by default. Customize `_run_aide_agent()` to use different metrics.

##Further Reading

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [GEPA GitHub](https://github.com/gepa-ai/gepa)
- [MLE-Dojo Documentation](https://github.com/mle-infrastructure/mle-dojo)

## Support

For issues specific to this integration, please check the main MLE-Dojo repository or GEPA repository.
