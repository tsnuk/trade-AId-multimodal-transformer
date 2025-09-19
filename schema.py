"""schema.py

Modern input schema system with dataclass, validation, and processing pipeline support.

This module provides a sophisticated configuration system that complements the programmatic
list-based input_schema with type-safe, validated, and extensible configurations.

Features:
- Type-safe dataclass definitions
- Built-in validation and error handling
- Processing pipeline architecture with sequential data flow
- YAML/JSON configuration support
- Compatibility with programmatic schemas
- Support for both built-in and external processing functions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import numbers
import os
import yaml
import json


@dataclass
class ProcessingStep:
    """Represents a single processing step in the pipeline"""
    function: str
    args: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        """Validate the processing step"""
        if not isinstance(self.function, str):
            raise TypeError(f"Processing function must be a string, got {type(self.function)}")
        if not isinstance(self.args, dict):
            raise TypeError(f"Processing args must be a dictionary, got {type(self.args)}")


@dataclass
class InputSchema:
    """
    Modern input schema matching the YAML structure with type safety, validation,
    and processing pipeline support.

    This complements the programmatic list-based input_schema with a sophisticated configuration
    system that supports complex data processing workflows with sequential data flow.

    Structure matches input_schemas.yaml format:
    - modality_name: str (required, moved to top)
    - path: str/Path (required)
    - column_number: int (required)
    - has_header: bool (required)
    - processing_steps: List[ProcessingStep] (optional, sequential pipeline)
    - cross_attention: bool (optional, default True)
    - randomness_size: Optional[int] (optional)
    """
    modality_name: str
    path: Union[str, Path]
    column_number: int
    has_header: bool = True
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    cross_attention: bool = True
    randomness_size: Optional[int] = None

    def __post_init__(self):
        """Validate the complete schema"""
        # Validate modality name
        if not self.modality_name or not isinstance(self.modality_name, str):
            raise ValueError("modality_name must be a non-empty string")

        # Convert and validate path
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

        # Validate column number
        if not isinstance(self.column_number, int) or self.column_number < 1:
            raise ValueError(f"column_number must be a positive integer, got {self.column_number}")

        # Validate has_header
        if not isinstance(self.has_header, bool):
            raise TypeError(f"has_header must be a boolean, got {type(self.has_header).__name__}")

        # Validate cross_attention
        if not (isinstance(self.cross_attention, bool) or self.cross_attention is None):
            raise TypeError(f"cross_attention must be a boolean or None, got {type(self.cross_attention).__name__}")

        # Validate processing steps
        for i, step in enumerate(self.processing_steps):
            if not isinstance(step, ProcessingStep):
                raise TypeError(f"Processing step {i} must be a ProcessingStep instance")

        # Validate randomness_size
        if self.randomness_size is not None:
            if not isinstance(self.randomness_size, int) or not (1 <= self.randomness_size <= 3):
                raise ValueError("randomness_size must be an integer between 1-3 or None")

    @classmethod
    def from_legacy_list(cls, legacy_list: List[Any], modality_name: str = "") -> 'InputSchema':
        """
        Create InputSchema from programmatic list format for compatibility.

        Legacy format: [Path, Col Num, Header, Percent Changes, Num Whole Digits,
                       Decimal Places, Bins, Rand Size, Cross-Attend, Modality Name]
        """
        if len(legacy_list) < 3:
            raise ValueError("Legacy list must have at least 3 elements (path, column, header)")

        # Extract processing parameters and convert to processing steps
        processing_steps = []

        # Percent changes
        if len(legacy_list) > 3 and legacy_list[3]:
            processing_steps.append(ProcessingStep(
                function='calculate_percent_changes',
                args={}
            ))

        # Range numeric data (num_whole_digits, decimal_places)
        if len(legacy_list) > 4 and (legacy_list[4] is not None or
                                    (len(legacy_list) > 5 and legacy_list[5] is not None)):
            args = {}
            if len(legacy_list) > 4 and legacy_list[4] is not None:
                args['num_whole_digits'] = legacy_list[4]
            if len(legacy_list) > 5 and legacy_list[5] is not None:
                args['decimal_places'] = legacy_list[5]

            processing_steps.append(ProcessingStep(
                function='range_numeric_data',
                args=args
            ))

        # Binning
        if len(legacy_list) > 6 and legacy_list[6] is not None:
            processing_steps.append(ProcessingStep(
                function='bin_numeric_data',
                args={'num_bins': legacy_list[6]}
            ))

        # Randomness (converted to randomness_size parameter, not processing step)
        randomness_size = None
        if len(legacy_list) > 7 and legacy_list[7] is not None:
            randomness_size = legacy_list[7]

        # Cross attention
        cross_attention = True
        if len(legacy_list) > 8 and legacy_list[8] is not None:
            cross_attention = legacy_list[8]

        # Schema name
        schema_name = modality_name
        if len(legacy_list) > 9 and legacy_list[9]:
            schema_name = legacy_list[9]
        elif not modality_name:
            schema_name = f"Legacy Schema {Path(legacy_list[0]).name}"

        return cls(
            modality_name=schema_name,
            path=legacy_list[0],
            column_number=legacy_list[1],
            has_header=legacy_list[2] if len(legacy_list) > 2 else True,
            processing_steps=processing_steps,
            cross_attention=cross_attention,
            randomness_size=randomness_size
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InputSchema':
        """Create InputSchema from dictionary configuration (from YAML/JSON)"""
        # Convert processing steps dictionaries to ProcessingStep objects
        processing_steps = []
        for step_dict in config_dict.get('processing_steps', []):
            processing_steps.append(ProcessingStep(**step_dict))

        return cls(
            modality_name=config_dict['modality_name'],
            path=config_dict['path'],
            column_number=config_dict['column_number'],
            has_header=config_dict.get('has_header', True),
            processing_steps=processing_steps,
            cross_attention=config_dict.get('cross_attention', True),
            randomness_size=config_dict.get('randomness_size')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert InputSchema to dictionary for serialization to YAML/JSON"""
        return {
            'modality_name': self.modality_name,
            'path': str(self.path),
            'column_number': self.column_number,
            'has_header': self.has_header,
            'processing_steps': [
                {
                    'function': step.function,
                    'args': step.args,
                    'enabled': step.enabled
                }
                for step in self.processing_steps
            ],
            'cross_attention': self.cross_attention,
            'randomness_size': self.randomness_size
        }

    def to_legacy_list(self) -> List[Any]:
        """Convert back to programmatic list format for compatibility"""
        # Initialize with required fields
        legacy_list = [
            str(self.path),
            self.column_number,
            self.has_header
        ]

        # Extract processing parameters from steps
        convert_to_percents = False
        num_whole_digits = None
        decimal_places = None
        num_bins = None

        for step in self.processing_steps:
            if step.function == 'calculate_percent_changes':
                convert_to_percents = True
            elif step.function == 'range_numeric_data':
                num_whole_digits = step.args.get('num_whole_digits')
                decimal_places = step.args.get('decimal_places')
            elif step.function == 'bin_numeric_data':
                num_bins = step.args.get('num_bins')

        # Extend list with processing parameters
        legacy_list.extend([
            convert_to_percents,
            num_whole_digits,
            decimal_places,
            num_bins,
            self.randomness_size,
            self.cross_attention,
            self.modality_name
        ])

        return legacy_list

    def validate(self) -> bool:
        """
        Comprehensive validation of the schema.

        Returns:
            True if valid, raises appropriate exceptions if invalid
        """
        # Validation is handled in __post_init__, but this method can be called
        # explicitly for re-validation after modifications

        # Import here to avoid circular imports
        from processing_registry import validate_function_exists, validate_function_arguments

        # Validate that all processing functions exist and have valid arguments
        for step in self.processing_steps:
            if step.enabled:
                if not validate_function_exists(step.function):
                    raise ImportError(f"Processing function '{step.function}' cannot be resolved")

                # Validate function arguments for built-in functions
                validate_function_arguments(step.function, step.args)

        return True


class SchemaManager:
    """Manages multiple InputSchema instances and provides utilities"""

    def __init__(self):
        self.schemas: List[InputSchema] = []

    def add_schema(self, schema: InputSchema):
        """Add a schema to the manager"""
        schema.validate()
        self.schemas.append(schema)

    def add_from_legacy_list(self, legacy_list: List[Any], modality_name: str = ""):
        """Add schema from programmatic list format"""
        schema = InputSchema.from_legacy_list(legacy_list, modality_name)
        self.add_schema(schema)

    def get_schema_by_name(self, name: str) -> Optional[InputSchema]:
        """Get schema by name"""
        for schema in self.schemas:
            if schema.modality_name == name:
                return schema
        return None

    def to_legacy_format(self) -> List[List[Any]]:
        """Convert all schemas back to programmatic format"""
        return [schema.to_legacy_list() for schema in self.schemas]

    def validate_all(self) -> bool:
        """Validate all schemas"""
        for schema in self.schemas:
            schema.validate()
        return True

    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save all schemas to YAML file"""
        config = {
            'modalities': [schema.to_dict() for schema in self.schemas]
        }

        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def load_from_yaml(self, file_path: Union[str, Path]) -> None:
        """Load schemas from YAML file"""
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        self.schemas = []
        for modality_config in config.get('modalities', []):
            schema = InputSchema.from_dict(modality_config)
            self.add_schema(schema)


# Backward compatibility utilities
def convert_legacy_input_schemas(num_schemas: int, globals_dict: dict) -> SchemaManager:
    """
    Convert legacy input_schema_1, input_schema_2, etc. to modern SchemaManager

    Args:
        num_schemas: Number of input schemas to look for
        globals_dict: The globals() dictionary to search in

    Returns:
        SchemaManager with converted schemas
    """
    manager = SchemaManager()

    for i in range(1, num_schemas + 1):
        schema_name = f'input_schema_{i}'
        if schema_name in globals_dict:
            legacy_list = globals_dict[schema_name]
            if legacy_list:  # Skip empty lists
                schema = InputSchema.from_legacy_list(legacy_list, f"Schema {i}")
                manager.add_schema(schema)

    return manager