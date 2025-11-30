"""
Configuration for the MLE-STAR agent.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import yaml
import os
from pathlib import Path

@dataclass
class MLEStarConfig:
    """Configuration for the MLE-STAR agent."""
    
    # LLM Configuration
    llm: Dict[str, Any] = field(
        default_factory=lambda: {
            "model_name": "gpt-4",
            "model_mode": "openai",
            "temperature": 0.7,
            "max_completion_tokens": 2048,
            "max_prompt_tokens": 4000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    )
    
    # Kaggle Configuration
    kaggle: Dict[str, Any] = field(
        default_factory=lambda: {
            "username": os.getenv("KAGGLE_USERNAME", ""),
            "key": os.getenv("KAGGLE_KEY", ""),
            "work_dir": "mle_star_workspace",
            "competition": "titanic",  # Default competition
            "timeout": 3600,  # 1 hour timeout for long-running operations
        }
    )
    
    # Search Configuration
    search: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "provider": "perplexity",  # or "google", "serpapi", etc.
            "max_results": 5,
            "cache_enabled": True,
            "cache_dir": "search_cache"
        }
    )
    
    # Feature Engineering Configuration
    feature_engineering: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "auto_feature_generation": True,
            "feature_selection": {
                "enabled": True,
                "method": "importance",  # or "correlation", "variance", etc.
                "max_features": 100
            },
            "feature_scaling": {
                "enabled": True,
                "method": "standard"  # or "minmax", "robust", etc.
            },
            "categorical_encoding": {
                "method": "onehot"  # or "target", "count", "embedding", etc.
            }
        }
    )
    
    # Model Configuration
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "ensemble",  # or "single"
            "base_models": [
                {"type": "xgboost", "params": {}},
                {"type": "lightgbm", "params": {}},
                {"type": "random_forest", "params": {}}
            ],
            "ensemble_method": "stacking",  # or "voting", "blending", etc.
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "bayesian",  # or "grid", "random", etc.
                "max_evals": 50,
                "patience": 10
            },
            "cross_validation": {
                "enabled": True,
                "folds": 5,
                "shuffle": True,
                "random_state": 42
            }
        }
    )
    
    # Training Configuration
    training: Dict[str, Any] = field(
        default_factory=lambda: ({
            "batch_size": 32,
            "epochs": 100,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "monitor": "val_loss",
                "mode": "min"
            },
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "auto",  # Will be set based on problem type
            "metrics": ["accuracy"]
        })
    )
    
    # Evaluation Configuration
    evaluation: Dict[str, Any] = field(
        default_factory=lambda: ({
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "threshold_optimization": {
                "enabled": True,
                "metric": "f1"
            },
            "interpretability": {
                "shap": True,
                "lime": False,
                "feature_importance": True
            },
            "error_analysis": {
                "enabled": True,
                "top_errors": 10
            }
        })
    )
    
    # Deployment Configuration
    deployment: Dict[str, Any] = field(
        default_factory=lambda: ({
            "format": "onnx",  # or "pickle", "tensorflow", "pytorch", etc.
            "api_type": "rest",  # or "grpc"
            "monitoring": {
                "enabled": True,
                "metrics": ["latency", "throughput", "error_rate"],
                "alerting": {
                    "enabled": True,
                    "slack_webhook": ""
                }
            },
            "scaling": {
                "enabled": True,
                "min_replicas": 1,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            }
        })
    )
    
    # Logging and Experiment Tracking
    logging: Dict[str, Any] = field(
        default_factory=lambda: ({
            "enabled": True,
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "mle_star.log",
            "experiment_tracking": {
                "enabled": True,
                "backend": "mlflow",  # or "wandb", "tensorboard", etc.
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "mle_star_experiment"
            }
        })
    )
    
    # Advanced Configuration
    advanced: Dict[str, Any] = field(
        default_factory=lambda: ({
            "parallel_processing": {
                "enabled": True,
                "max_workers": 4,
                "backend": "joblib"  # or "ray", "dask", etc.
            },
            "caching": {
                "enabled": True,
                "directory": ".mle_star_cache",
                "ttl": 86400  # 24 hours in seconds
            },
            "debug": {
                "enabled": False,
                "log_level": "DEBUG",
                "save_intermediate": True
            },
            "security": {
                "encrypt_secrets": True,
                "vault_url": ""
            }
        })
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return asdict(self)
    
    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """Save the configuration to a YAML file."""
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLEStarConfig':
        """Create a configuration from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'MLEStarConfig':
        """Load a configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update the configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate the configuration.
        
        Returns:
            List of validation errors, empty if valid.
        """
        errors = []
        
        # Validate LLM configuration
        if not self.llm.get("model_name"):
            errors.append("LLM model_name is required")
        
        # Validate Kaggle configuration if Kaggle is being used
        if self.kaggle.get("enabled", False):
            if not self.kaggle.get("username") or not self.kaggle.get("key"):
                errors.append("Kaggle username and key are required when Kaggle is enabled")
        
        # Validate search configuration if search is enabled
        if self.search.get("enabled", False) and not self.search.get("provider"):
            errors.append("Search provider is required when search is enabled")
        
        # Add more validation rules as needed
        
        return errors

# Default configuration
DEFAULT_CONFIG = MLEStarConfig()

def load_config(config_path: Optional[str] = None) -> MLEStarConfig:
    """Load configuration from a file or use defaults.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        MLEStarConfig: The loaded configuration.
    """
    if config_path and os.path.exists(config_path):
        return MLEStarConfig.from_yaml(config_path)
    
    # Try to load from default locations
    default_paths = [
        os.path.join(os.getcwd(), "mle_star_config.yaml"),
        os.path.join(os.path.expanduser("~"), ".config", "mle_star", "config.yaml"),
        "/etc/mle_star/config.yaml"
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return MLEStarConfig.from_yaml(path)
    
    # Return default config if no config file is found
    return DEFAULT_CONFIG
