import json
import os
from typing import Dict, Any

class ConfigManager:
    """Statically loads external JSON configurations to eliminate hardcoded parameters."""
    _config: Dict[str, Any] = {}

    @classmethod
    def load(cls, config_path: str = "config/settings.json") -> Dict[str, Any]:
        if not cls._config:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Missing critical configuration file: {config_path}")
            with open(config_path, 'r') as f:
                cls._config = json.load(f)
        return cls._config

    @classmethod
    def get_optuna_space(cls) -> Dict[str, Any]:
        return cls.load()["hyperparameters"]["optuna_search_space"]
        
    @classmethod
    def get_training_params(cls) -> Dict[str, Any]:
        return cls.load()["hyperparameters"]["training"]