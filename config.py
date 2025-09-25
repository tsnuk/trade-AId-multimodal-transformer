"""config.py

Legacy configuration support for programmatic input schemas.

This module provides backward compatibility for programmatic configuration
when YAML configuration files are not present. Contains input schema definitions
and conditionally loads hyperparameters when config.yaml is missing.
"""

from pathlib import Path


# Check if YAML configuration exists
_yaml_config_exists = (Path('input_schemas.yaml').exists() and Path('config.yaml').exists())

# Export variables - conditionally include hyperparameters if YAML config is missing
__all__ = [
    # Programmatic input schemas (always available)
    'num_input_schemas', 'input_schema_1', 'input_schema_2', 'input_schema_3', 'input_schema_4',
    'input_schema_5', 'input_schema_6', 'input_schema_7', 'input_schema_8', 'input_schema_9', 'input_schema_10',
    # Model-specific constants
    'fixed_values'
]

# Add hyperparameters to exports only if YAML config doesn't exist
if not _yaml_config_exists:
    __all__.extend([
        # Training hyperparameters
        'batch_size', 'block_size', 'max_iters', 'eval_interval', 'eval_iters', 'learning_rate', 'device',
        # Model architecture
        'n_embd', 'n_head', 'n_layer', 'dropout',
        # File paths and settings
        'project_file_path', 'model_file_name', 'output_file_name',
        # Data splitting
        'validation_size', 'num_validation_files',
        # Model management
        'create_new_model', 'save_model'
    ])

# Conditional hyperparameter loading - only define these if YAML config doesn't exist
if not _yaml_config_exists:
    # Import modules only when needed (lazy imports)
    import torch
    from datetime import datetime

    # Training hyperparameters
    batch_size = 8
    block_size = 6
    max_iters = 20000
    eval_interval = 50
    eval_iters = 40
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model architecture
    n_embd = 16
    n_head = 4
    n_layer = 4
    dropout = 0.2

    # File paths and settings
    project_file_path = './'
    model_file_name = project_file_path + 'output/' + 'TransformerModel.pth'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f'output_run_{timestamp}.txt'

    # Data splitting
    validation_size = 0.1
    num_validation_files = 0

    # Model management
    create_new_model = 0
    save_model = 1

# Programmatic Input Schema Definitions
# For detailed configuration guide, see CONFIGURATION_GUIDE.md

# Number of input schemas available for programmatic configuration
num_input_schemas = 10

# Input schema structure: [Path, Col Num, Header, Percent Changes, Num Whole Digits, Decimal Places, Bins, Rand Size, Cross-Attend, Modality Name]
# Types: [(str), (int), (bool), (bool or None), (int or None), (int or None), (int or None), (int or None), (bool or None), (str or None)]
# Updated for local Windows environment #
input_schema_1 = ['./data_1/tick_10m/', 13, True, False, 2, 1, None, None, True, '200 stocks']
input_schema_2 = ['./data_1/tick_10m/', 13, True, True, None, 2, 6, None, False, '200 stocks - percents']
input_schema_3 = ['./data_1/tick_10m/', 9, True, False, None, None, None, None, False, 'Time']
input_schema_4 = ['./data_1/tick_10m/', 5, True, False, None, None, None, None, False, 'Day of week']
input_schema_5 = []
input_schema_6 = []
input_schema_7 = []
input_schema_8 = []
input_schema_9 = []
input_schema_10 = []

# Fixed values for custom embedding layer
fixed_values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
# when creating the embedding table, each element of the embedding vectors is randomly selected from this fixed_values list