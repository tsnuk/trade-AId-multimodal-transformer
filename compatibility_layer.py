"""compatibility_layer.py

Configuration compatibility layer for seamless integration between YAML and programmatic systems.

This module provides transparent compatibility between the programmatic list-based input_schema
system and the YAML-based configuration system. It enables users to choose the configuration
method that best fits their workflow without any code changes.

Features:
- Automatic detection of configuration method (YAML vs programmatic)
- Transparent conversion between programmatic lists and YAML schemas
- Integration with existing global variables and patterns
- Zero changes required to switch between methods
- Support for both standard users and advanced automation needs
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

from config_manager import ConfigManager
from processing_pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


class CompatibilityMode:
    """Manages compatibility between YAML and programmatic configuration systems"""

    def __init__(self):
        self.mode = None
        self.config_manager = None
        self.legacy_schemas = []
        self.is_initialized = False

    def detect_and_initialize(self, globals_dict: dict) -> str:
        """
        Detect whether to use programmatic or YAML configuration system and initialize accordingly.

        Args:
            globals_dict: The globals() dictionary from the calling module

        Returns:
            'legacy' (programmatic) or 'modern' (YAML) indicating which system is being used
        """
        if self.is_initialized:
            return self.mode

        yaml_config_exists = (
            Path('input_schemas.yaml').exists() and
            Path('config.yaml').exists()
        )

        programmatic_schemas_exist = any(
            key.startswith('input_schema_') and globals_dict.get(key)
            for key in globals_dict.keys()
        )

        if yaml_config_exists:
            self.mode = 'modern'
            logger.info("YAML configuration system detected")
            self._initialize_modern_system()
        elif programmatic_schemas_exist:
            self.mode = 'legacy'
            logger.info("Programmatic configuration system detected")
            self._initialize_legacy_system(globals_dict)
        else:
            self.mode = 'legacy'
            logger.warning("No configuration detected, defaulting to programmatic mode")

        self.is_initialized = True
        return self.mode

    def _initialize_modern_system(self):
        """Initialize the YAML-based configuration system"""
        try:
            self.config_manager = ConfigManager()
            self.config_manager.load_all_configs()
            logger.info(f"YAML system initialized with {len(self.config_manager.schema_manager.schemas)} modalities")
        except Exception as e:
            logger.error(f"Failed to initialize YAML system: {e}")
            self.mode = 'legacy'
            self.config_manager = None

    def _initialize_legacy_system(self, globals_dict: dict):
        """Initialize using programmatic input schemas from globals"""
        try:
            from config import num_input_schemas

            self.legacy_schemas = []
            for i in range(1, num_input_schemas + 1):
                schema_name = f'input_schema_{i}'
                if schema_name in globals_dict and globals_dict[schema_name]:
                    self.legacy_schemas.append(globals_dict[schema_name])

            logger.info(f"Programmatic system initialized with {len(self.legacy_schemas)} input schemas")
        except Exception as e:
            logger.error(f"Failed to initialize programmatic system: {e}")
            self.legacy_schemas = []

    def get_all_modality_params(self) -> List[List[Any]]:
        """
        Get modality parameters in the format expected by existing code.

        Returns:
            List of parameter lists compatible with existing code
        """
        if self.mode == 'modern' and self.config_manager:
            return [schema.to_legacy_list() for schema in self.config_manager.schema_manager.schemas]
        else:
            return self.legacy_schemas

    def get_system_parameters(self) -> Dict[str, Any]:
        """
        Get system parameters (hyperparameters) for the current configuration mode.

        Returns:
            Dictionary of system parameters compatible with existing code
        """
        if self.mode == 'modern' and self.config_manager and self.config_manager.system_config:
            sys_config = self.config_manager.system_config
            import torch

            device = sys_config.device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            return {
                'batch_size': sys_config.batch_size,
                'block_size': sys_config.block_size,
                'max_iters': sys_config.max_iters,
                'eval_interval': sys_config.eval_interval,
                'eval_iters': sys_config.eval_iters,
                'learning_rate': sys_config.learning_rate,
                'device': device,
                'n_embd': sys_config.n_embd,
                'n_head': sys_config.n_head,
                'n_layer': sys_config.n_layer,
                'dropout': sys_config.dropout,
                'validation_size': sys_config.validation_size,
                'num_validation_files': sys_config.num_validation_files,
                'create_new_model': sys_config.create_new_model,
                'save_model': sys_config.save_model,
                'model_file_name': sys_config.model_file_name,
                'project_file_path': sys_config.project_file_path,
                'output_file_name': sys_config.output_file_name,
                'fixed_values': sys_config.fixed_values
            }
        else:
            from config import (
                batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device,
                n_embd, n_head, n_layer, dropout, validation_size, num_validation_files,
                create_new_model, save_model, model_file_name, project_file_path, output_file_name, fixed_values
            )
            return {
                'batch_size': batch_size,
                'block_size': block_size,
                'max_iters': max_iters,
                'eval_interval': eval_interval,
                'eval_iters': eval_iters,
                'learning_rate': learning_rate,
                'device': device,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'dropout': dropout,
                'validation_size': validation_size,
                'num_validation_files': num_validation_files,
                'create_new_model': create_new_model,
                'save_model': save_model,
                'model_file_name': model_file_name,
                'project_file_path': project_file_path,
                'output_file_name': output_file_name,
                'fixed_values': fixed_values
            }

    def process_modality_data(self, modality_index: int, raw_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Process modality data using either YAML pipeline or programmatic processing.

        Args:
            modality_index: Index of the modality (0-based)
            raw_data: Raw data to process

        Returns:
            Tuple of (processed_data, metadata)
        """
        if self.mode == 'modern' and self.config_manager:
            schemas = self.config_manager.schema_manager.schemas
            if modality_index < len(schemas):
                schema = schemas[modality_index]
                pipeline = ProcessingPipeline()
                result = pipeline.execute_for_schema(raw_data, schema)

                if result.success:
                    return result.processed_data, result.metadata
                else:
                    logger.error(f"Modern pipeline failed for modality {modality_index}: {result.error}")
                    return raw_data, {'error': result.error}
            else:
                logger.warning(f"Modality index {modality_index} out of range")
                return raw_data, {}
        else:
            return raw_data, {}

    def get_modality_metadata(self, modality_index: int) -> Dict[str, Any]:
        """
        Get metadata for a specific modality.

        Args:
            modality_index: Index of the modality (0-based)

        Returns:
            Dictionary containing modality metadata
        """
        if self.mode == 'modern' and self.config_manager:
            schemas = self.config_manager.schema_manager.schemas
            if modality_index < len(schemas):
                schema = schemas[modality_index]
                return {
                    'modality_name': schema.modality_name,
                    'cross_attention': schema.cross_attention,
                    'randomness_size': schema.randomness_size,
                    'processing_steps_count': len(schema.processing_steps),
                    'mode': 'modern'
                }

        if modality_index < len(self.legacy_schemas):
            programmatic_schema = self.legacy_schemas[modality_index]
            return {
                'modality_name': programmatic_schema[9] if len(programmatic_schema) > 9 else f'Modality {modality_index + 1}',
                'cross_attention': programmatic_schema[8] if len(programmatic_schema) > 8 else False,
                'randomness_size': programmatic_schema[7] if len(programmatic_schema) > 7 else None,
                'processing_steps_count': 0,
                'mode': 'programmatic'
            }

        return {'mode': self.mode}

    def is_percent_modality(self, modality_index: int) -> bool:
        """
        Check if a modality uses percentage conversion.
        This is needed for special handling in directional success calculations.

        Args:
            modality_index: Index of the modality (0-based)

        Returns:
            True if modality converts to percentages
        """
        if self.mode == 'modern' and self.config_manager:
            schemas = self.config_manager.schema_manager.schemas
            if modality_index < len(schemas):
                schema = schemas[modality_index]
                for step in schema.processing_steps:
                    if step.function == 'convert_to_percent_changes' and step.enabled:
                        return True
                return False
        else:
            if modality_index < len(self.legacy_schemas):
                programmatic_schema = self.legacy_schemas[modality_index]
                return len(programmatic_schema) > 3 and programmatic_schema[3]
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration for debugging/logging"""
        summary = {
            'mode': self.mode,
            'initialized': self.is_initialized,
            'modalities_count': 0
        }

        if self.mode == 'modern' and self.config_manager:
            summary.update({
                'modalities_count': len(self.config_manager.schema_manager.schemas),
                'yaml_configs_loaded': True,
                'system_config_loaded': self.config_manager.system_config is not None
            })
        else:
            summary.update({
                'modalities_count': len(self.legacy_schemas),
                'yaml_configs_loaded': False,
                'system_config_loaded': False
            })

        return summary


compatibility_layer = CompatibilityMode()


def initialize_compatibility_layer(globals_dict: dict) -> str:
    """
    Initialize the compatibility layer with automatic mode detection.

    This function should be called early in the application startup to detect
    and initialize the appropriate configuration system.

    Args:
        globals_dict: The globals() dictionary from the calling module

    Returns:
        'legacy' (programmatic) or 'modern' (YAML) indicating which system is being used
    """
    return compatibility_layer.detect_and_initialize(globals_dict)


def get_modality_parameters() -> List[List[Any]]:
    """
    Get all modality parameters in the format expected by existing code.

    Returns:
        List of modality parameter lists compatible with existing code
    """
    return compatibility_layer.get_all_modality_params()


def get_system_configuration() -> Dict[str, Any]:
    """
    Get system configuration parameters for existing code compatibility.

    Returns:
        Dictionary of system parameters
    """
    if not compatibility_layer.is_initialized:
        compatibility_layer.detect_and_initialize(globals())
    return compatibility_layer.get_system_parameters()


def is_modern_mode() -> bool:
    """Check if the system is running in YAML configuration mode"""
    return compatibility_layer.mode == 'modern'


def is_legacy_mode() -> bool:
    """Check if the system is running in programmatic configuration mode"""
    return compatibility_layer.mode == 'legacy'