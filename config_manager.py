"""config_manager.py

Configuration management system for YAML-based configuration loading, validation, and saving.

This module provides centralized configuration management that handles both
input schemas (modality configurations) and system settings (training parameters,
model architecture, etc.) from YAML files with comprehensive validation.

Features:
- YAML configuration loading and saving
- Startup validation of all configurations
- Integration with programmatic configuration systems
- Comprehensive error handling and reporting
- Support for external function validation
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass

from schema import SchemaManager
from processing_registry import validate_function_exists

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System configuration matching config.yaml structure"""
    # Project settings
    project_file_path: str
    output_file_name: str
    model_file_name: str
    create_new_model: bool
    save_model: bool
    device: str

    # Data splitting
    validation_size: float
    num_validation_files: int

    # Training parameters
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    learning_rate: float

    # Model architecture
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    fixed_values: List[float]

    def __post_init__(self):
        """Validate system configuration."""
        project_path = Path(self.project_file_path)
        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")

        if not 0.0 <= self.validation_size <= 1.0:
            raise ValueError(f"validation_size must be between 0.0 and 1.0, got {self.validation_size}")
        if self.num_validation_files < 0:
            raise ValueError("num_validation_files must be non-negative")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")

        if not isinstance(self.fixed_values, list) or not self.fixed_values:
            raise ValueError("fixed_values must be a non-empty list")
        for i, val in enumerate(self.fixed_values):
            if not isinstance(val, (int, float)):
                raise ValueError(f"fixed_values[{i}] must be a number, got {type(val).__name__}")

        if self.device not in ['cpu', 'cuda', 'auto']:
            logger.warning(f"Device '{self.device}' may not be supported. Common values: 'cpu', 'cuda', 'auto'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary.

        Args:
            config_dict: Dictionary loaded from YAML configuration.

        Returns:
            SystemConfig instance created from dictionary.
        """
        flattened = {}

        project_settings = config_dict.get('project_settings', {})
        flattened.update({
            'project_file_path': project_settings.get('project_file_path', ''),
            'output_file_name': project_settings.get('output_file_name', 'training_log.txt'),
            'model_file_name': project_settings.get('model_file_name', 'model.pth'),
            'create_new_model': bool(project_settings.get('create_new_model', 1)),
            'save_model': bool(project_settings.get('save_model', 1)),
            'device': project_settings.get('device', 'cpu')
        })

        data_splitting = config_dict.get('data_splitting', {})
        flattened.update({
            'validation_size': data_splitting.get('validation_size', 0.1),
            'num_validation_files': data_splitting.get('num_validation_files', 0)
        })

        training_params = config_dict.get('training_parameters', {})
        flattened.update({
            'batch_size': training_params.get('batch_size', 32),
            'block_size': training_params.get('block_size', 64),
            'max_iters': training_params.get('max_iters', 5000),
            'eval_interval': training_params.get('eval_interval', 500),
            'eval_iters': training_params.get('eval_iters', 40),
            'learning_rate': training_params.get('learning_rate', 3e-4)
        })

        model_arch = config_dict.get('model_architecture', {})
        flattened.update({
            'n_embd': model_arch.get('n_embd', 384),
            'n_head': model_arch.get('n_head', 6),
            'n_layer': model_arch.get('n_layer', 6),
            'dropout': model_arch.get('dropout', 0.2),
            'fixed_values': model_arch.get('fixed_values', [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5])
        })

        return cls(**flattened)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SystemConfig back to nested dictionary structure.

        Returns:
            Dictionary with nested configuration structure.
        """
        return {
            'project_settings': {
                'project_file_path': self.project_file_path,
                'output_file_name': self.output_file_name,
                'model_file_name': self.model_file_name,
                'create_new_model': int(self.create_new_model),
                'save_model': int(self.save_model),
                'device': self.device
            },
            'data_splitting': {
                'validation_size': self.validation_size,
                'num_validation_files': self.num_validation_files
            },
            'training_parameters': {
                'batch_size': self.batch_size,
                'block_size': self.block_size,
                'max_iters': self.max_iters,
                'eval_interval': self.eval_interval,
                'eval_iters': self.eval_iters,
                'learning_rate': self.learning_rate
            },
            'model_architecture': {
                'n_embd': self.n_embd,
                'n_head': self.n_head,
                'n_layer': self.n_layer,
                'dropout': self.dropout,
                'fixed_values': self.fixed_values
            }
        }


class ConfigManager:
    """Centralized configuration manager for the entire system.

    Handles loading, validation, and management of both input schemas
    and system configurations from YAML files.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files.
                       If None, uses current working directory.
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.schema_manager = SchemaManager()
        self.system_config: Optional[SystemConfig] = None

        self.input_schemas_path = self.config_dir / 'input_schemas.yaml'
        self.system_config_path = self.config_dir / 'config.yaml'

    def load_all_configs(self) -> None:
        """Load all configuration files and validate them.

        This method should be called during application startup to ensure
        all configurations are valid before proceeding.
        """
        logger.info("Loading all configuration files...")

        try:
            self.load_system_config()
            logger.info("System configuration loaded successfully")

            self.load_input_schemas()
            logger.info(f"Input schemas loaded successfully ({len(self.schema_manager.schemas)} modalities)")

            self.validate_all_functions()
            logger.info("All processing functions validated successfully")

            logger.info("Configuration loading completed successfully")

        except Exception as e:
            error_msg = f"Configuration loading failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def load_system_config(self, file_path: Optional[Union[str, Path]] = None) -> SystemConfig:
        """Load system configuration from YAML file.

        Args:
            file_path: Path to config.yaml file. If None, uses default path.

        Returns:
            Loaded and validated SystemConfig.

        Raises:
            FileNotFoundError: If config file not found.
            ValueError: If YAML is invalid.
            RuntimeError: If loading fails.
        """
        config_path = Path(file_path) if file_path else self.system_config_path

        if not config_path.exists():
            raise FileNotFoundError(f"System config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            self.system_config = SystemConfig.from_dict(config_data)
            return self.system_config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in system config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load system config: {e}")

    def load_input_schemas(self, file_path: Optional[Union[str, Path]] = None) -> SchemaManager:
        """Load input schemas from YAML file.

        Args:
            file_path: Path to input_schemas.yaml file. If None, uses default path.

        Returns:
            Loaded and validated SchemaManager.

        Raises:
            FileNotFoundError: If schemas file not found.
            ValueError: If YAML is invalid.
            RuntimeError: If loading fails.
        """
        schemas_path = Path(file_path) if file_path else self.input_schemas_path

        if not schemas_path.exists():
            raise FileNotFoundError(f"Input schemas file not found: {schemas_path}")

        try:
            self.schema_manager.load_from_yaml(schemas_path)
            return self.schema_manager

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in input schemas file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load input schemas: {e}")

    def save_system_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save system configuration to YAML file.

        Args:
            file_path: Path to output file. If None, uses default path.

        Raises:
            RuntimeError: If no configuration loaded or saving fails.
        """
        if not self.system_config:
            raise RuntimeError("No system configuration loaded to save")

        config_path = Path(file_path) if file_path else self.system_config_path

        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.system_config.to_dict(), f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise RuntimeError(f"Failed to save system config: {e}")

    def save_input_schemas(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save input schemas to YAML file.

        Args:
            file_path: Path to output file. If None, uses default path.

        Raises:
            RuntimeError: If saving fails.
        """
        schemas_path = Path(file_path) if file_path else self.input_schemas_path

        try:
            self.schema_manager.save_to_yaml(schemas_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save input schemas: {e}")

    def validate_all_functions(self) -> None:
        """Validate that all processing functions in schemas can be resolved.

        Raises:
            ImportError: If any function cannot be resolved.
        """
        errors = []

        for schema in self.schema_manager.schemas:
            for step in schema.processing_steps:
                if step.enabled and not validate_function_exists(step.function):
                    errors.append(f"Modality '{schema.modality_name}': Function '{step.function}' cannot be resolved")

        if errors:
            error_msg = "Function validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ImportError(error_msg)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded configurations.

        Returns:
            Dictionary containing configuration summary.
        """
        summary = {
            'system_config_loaded': self.system_config is not None,
            'input_schemas_loaded': len(self.schema_manager.schemas) > 0,
            'total_modalities': len(self.schema_manager.schemas),
            'config_files': {
                'system_config_path': str(self.system_config_path),
                'input_schemas_path': str(self.input_schemas_path),
                'system_config_exists': self.system_config_path.exists(),
                'input_schemas_exists': self.input_schemas_path.exists()
            }
        }

        if self.system_config:
            summary['system_config'] = {
                'device': self.system_config.device,
                'batch_size': self.system_config.batch_size,
                'max_iters': self.system_config.max_iters,
                'n_embd': self.system_config.n_embd,
                'n_head': self.system_config.n_head,
                'n_layer': self.system_config.n_layer,
                'fixed_values': len(self.system_config.fixed_values)
            }

        if self.schema_manager.schemas:
            summary['modalities'] = [
                {
                    'name': schema.modality_name,
                    'processing_steps': len(schema.processing_steps),
                    'cross_attention': schema.cross_attention
                }
                for schema in self.schema_manager.schemas
            ]

        return summary

    def create_example_configs(self) -> None:
        """Create example configuration files if they don't exist."""
        pass


config_manager = ConfigManager()


def load_configurations(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Convenience function to load all configurations.

    Args:
        config_dir: Directory containing configuration files.

    Returns:
        Initialized and loaded ConfigManager.
    """
    global config_manager
    if config_dir:
        config_manager = ConfigManager(config_dir)

    config_manager.load_all_configs()
    return config_manager


def get_system_config() -> SystemConfig:
    """Get the loaded system configuration.

    Returns:
        Loaded SystemConfig instance.

    Raises:
        RuntimeError: If system configuration not loaded.
    """
    if not config_manager.system_config:
        raise RuntimeError("System configuration not loaded. Call load_configurations() first.")
    return config_manager.system_config


def get_schema_manager() -> SchemaManager:
    """Get the loaded schema manager.

    Returns:
        SchemaManager instance.
    """
    return config_manager.schema_manager