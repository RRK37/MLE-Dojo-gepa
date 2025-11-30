# MLE-STAR Agent

Machine Learning Engineering via Search and Targeted Refinement (MLE-STAR) Agent extends AIDE with advanced capabilities for automated ML solution development.

## Features

- **Web Search Integration**: Uses Perplexity API to search for state-of-the-art models and solutions
- **Targeted Refinement**: Performs ablation studies to identify high-impact components
- **Code Block Refinement**: Iteratively improves specific code blocks based on ablation results
- **Ensemble Strategies**: Intelligently combines multiple solutions
- **Data Leakage Detection**: Automatically detects and fixes data leakage issues
- **Data Usage Validation**: Ensures all provided information is utilized

## Workflow Phases

1. **Search Phase**: Web search for effective models using Perplexity
2. **Foundation Phase**: Generate and merge initial solutions
3. **Refinement Phase**: Ablation studies and targeted code block improvements
4. **Ensemble Phase**: Combine multiple solutions
5. **Validation Phase**: Check for data leakage and ensure complete data usage

## Configuration

### Basic Configuration

```yaml
agent_type: "mlestar"

mlestar:
  search_iterations: 3
  refinement_iterations: 5
  perplexity_api_key: null  # or set PERPLEXITY_API_KEY env var
```

### Environment Variables

- `PERPLEXITY_API_KEY`: Perplexity API key for web search (required for search phase)
- `ANTHROPIC_API_KEY`: Claude API key (for LLM calls)

## Usage

```bash
python main.py --config config_mlestar.yaml
```

Or with command line arguments:

```bash
python main.py --agent-type mlestar --competition-name titanic --data-dir data/titanic --output-dir output
```

## MLE-STAR Prompts

The agent implements all 14 MLE-STAR prompts:

1. **Model Retrieval**: Search for effective models
2. **Initial Solution**: Generate solution from model description
3. **Solution Merging**: Integrate multiple solutions
4. **Ablation Study**: Identify high-impact components
5. **Ablation Summarization**: Summarize ablation results
6. **Refinement Planning**: Extract code blocks and plan improvements
7. **Code Refinement**: Implement improvements
8. **Alternative Refinement**: Suggest new plans when previous fail
9. **Ensemble Planning**: Plan ensemble strategies
10. **Ensemble Implementation**: Create ensemble solutions
11. **Error Debugging**: Fix code errors
12. **Data Leakage Detection**: Check for leakage
13. **Data Leakage Fix**: Correct leakage issues
14. **Data Usage Check**: Ensure all information is used

## Extending AIDE

MLE-STAR extends the AIDE agent, inheriting all AIDE functionality while adding:
- Perplexity-based web search
- Structured refinement workflow
- Ablation study capabilities
- Enhanced ensemble strategies

All AIDE features (drafting, improving, debugging) remain available as fallback.

