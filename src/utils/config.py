"""
Configuration management utilities.
Loads and validates configuration from YAML files.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "default.yaml"
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., "model.path")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration section."""
        return self._config.get(key, {})
    
    @property
    def model_path(self) -> str:
        """Get model path."""
        return self.get("model.path", "src/models/efficientNet_v1.pth")
    
    @property
    def backbone_name(self) -> str:
        """Get backbone name."""
        return self.get("model.backbone", "efficientnet_b0")
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self.get("model.num_classes", 5)
    
    @property
    def class_names(self) -> list:
        """Get class names."""
        return self.get("classes.names", ["Select", "Choice", "Prime", "Wagyu", "Japanese A5"])
    
    @property
    def project_root(self) -> Path:
        """Get project root path."""
        root = self.get("paths.project_root", ".")
        return Path(root).resolve()
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute using project root."""
        return self.project_root / relative_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
    
    Returns:
        Config object
    """
    return Config(config_path)

