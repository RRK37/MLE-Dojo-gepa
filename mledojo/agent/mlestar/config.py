"""
MLE-STAR Configuration - Supports both YAML and Python config.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import os


@dataclass
class MLEStarConfig:
    """MLE-STAR specific configuration."""
    # Algorithm parameters
    search_iterations: int = 3
    refinement_iterations: int = 5
    inner_refinement_steps: int = 3
    ensemble_iterations: int = 3
    num_models_to_retrieve: int = 3
    subsample_size: int = 30000
    
    # Web search
    perplexity_api_key: Optional[str] = field(default_factory=lambda: os.getenv('PERPLEXITY_API_KEY'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'search_iterations': self.search_iterations,
            'refinement_iterations': self.refinement_iterations,
            'inner_refinement_steps': self.inner_refinement_steps,
            'ensemble_iterations': self.ensemble_iterations,
            'num_models_to_retrieve': self.num_models_to_retrieve,
            'subsample_size': self.subsample_size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MLEStarConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'MLEStarConfig':
        """Load from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        mlestar_section = data.get('mlestar', {})
        return cls.from_dict(mlestar_section)


def load_config(config_path: Optional[str] = None) -> MLEStarConfig:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        return MLEStarConfig.from_yaml(config_path)
    return MLEStarConfig()
