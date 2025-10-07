# MULTIMODAL TRANSFORMER TRAINING FLOW

## Complete Program Execution Flow

```
START: main.py execution
|
|- 1. CONFIGURATION LOADING (main.py:29-64)
|   |- config_manager.py -> load_config()
|   |   |- Loads config.yaml -> system settings
|   |   '- Loads input_schemas.yaml -> modality configurations
|   |- schema.py -> SchemaManager.load_from_yaml()
|   |   |- Validates modality configurations
|   |   '- Creates InputSchema objects
|   '- Sets global config variables
|
|- 2. DATA LOADING & PROCESSING (main.py:66-259)
|   |- For each modality:
|   |   |- file_cache.py -> load_file_data_cached()
|   |   |   |- Loads CSV/TXT files from path
|   |   |   |- Extracts specified column
|   |   |   '- Returns raw numeric data + file info
|   |   |- data_utils.py -> Processing functions applied inline:
|   |   |   |- convert_to_percent_changes() (if enabled)
|   |   |   |- range_numeric_data() (if enabled)
|   |   |   '- bin_numeric_data() (if enabled)
|   |   '- data_utils.py -> numerical_representation()
|   |       '- Creates vocabulary from processed data
|   '- VOCABULARY BUILDING output (main.py:271-325)
|
|- 3. DATASET SPLITTING (main.py:334-378)
|   |- data_utils.py -> create_train_val_datasets()
|   |   |- If num_validation_files > 0: File-based split
|   |   '- Else: Percentage-based split using validation_size
|   |- file_cache.py -> cleanup_cache() (main.py:380)
|   '- DATASET SPLITTING output
|
|- 4. MODEL INITIALIZATION (main.py:451-485)
|   |- model.py -> MultimodalTransformer()
|   |   |- Creates embedding layers for each modality
|   |   |- Initializes MultimodalBlock layers
|   |   |   |- MultiHeadAttention (self-attention)
|   |   |   |- CrossAttention (cross-modal attention)
|   |   |   '- FeedForward layers
|   |   '- MultimodalPreBlock/PostBlock
|   |- If create_new_model=0: Load existing model weights
|   '- Move model to device (CPU/CUDA)
|
|- 5. TRAINING SETUP (main.py:487-592)
|   |- Creates PyTorch optimizer (AdamW)
|   |- Sets up training log file
|   |- data_utils.py -> write_initial_run_details()
|   |   '- Writes configuration summary to log
|   '- TRAINING STARTUP output
|
|- 6. MAIN TRAINING LOOP (main.py:598-653)
|   |
|   |- For each iteration (0 to max_iters):
|   |   |
|   |   |- EVALUATION (every eval_interval steps)
|   |   |   |- training_utils.py -> estimate_loss()
|   |   |   |   |- Runs model on train/val batches
|   |   |   |   |- Calculates train/validation loss
|   |   |   |   |- training_utils.py -> calculate_evaluation_metrics()
|   |   |   |   |   |- Calculates directional accuracy
|   |   |   |   |   '- Returns metrics dictionary
|   |   |   |   '- Returns average losses
|   |   |   |- LOSS METRICS output (console)
|   |   |   |- STEP log entry (training_log.txt)
|   |   |   |- Early stopping check (if validation improves)
|   |   |   '- Model saving (if save_model=1)
|   |   |
|   |   |- TRAINING STEP
|   |   |   |- training_utils.py -> get_batch('train')
|   |   |   |   '- Gets random training batches
|   |   |   |- model.forward() -> MultimodalTransformer
|   |   |   |   |- Embedding lookup for each modality
|   |   |   |   |- Processes through transformer blocks:
|   |   |   |   |   |- Self-attention (all modalities)
|   |   |   |   |   |- Cross-attention (if enabled)
|   |   |   |   |   '- FeedForward
|   |   |   |   '- Output predictions + loss
|   |   |   |- Calculates total loss (sum of modality losses)
|   |   |   |- Backpropagation
|   |   |   '- Optimizer step
|   |   |
|   |   '- Continue to next iteration
|   |
|   '- Loop continues until max_iters reached or early stopping
|
|- 7. TRAINING COMPLETION (main.py:655-668)
|   |- Final model save (if save_model=1)
|   '- TRAINING COMPLETE output
|
'- END: Program termination
```

## Key Files and Their Roles

| File | Primary Responsibility |
|------|----------------------|
| **main.py** | Main execution orchestrator |
| **compatibility_layer.py** | YAML/programmatic config bridging |
| **config_manager.py** | Configuration loading/validation |
| **schema.py** | YAML schema management |
| **data_utils.py** | Data loading, processing, splitting |
| **processing_pipeline.py** | Data processing coordination |
| **processing_registry.py** | Function validation/resolution |
| **model.py** | Neural network architecture |
| **training_utils.py** | Training/evaluation operations |
| **file_cache.py** | File caching and cleanup |

## Configuration Files

| File | Purpose |
|------|---------|
| **config.yaml** | System settings & hyperparameters |
| **input_schemas.yaml** | Modality definitions & processing |

## Output Files

| File | Content |
|------|---------|
| **output_file_name** (configurable) | Detailed training log |
| **model_file_name** (configurable) | Saved model weights |

## Key Processing Phases

### Phase 1: Configuration & Validation
- Load YAML configurations
- Validate modality settings
- Set up global parameters

### Phase 2: Data Pipeline
- Load CSV/TXT data files
- Apply processing functions (percentages, ranging, binning)
- Build vocabularies
- Split into train/validation sets

### Phase 3: Model Setup
- Initialize transformer architecture
- Set up self-attention and cross-attention layers
- Configure optimizer and training parameters

### Phase 4: Training Loop
- Training steps with backpropagation
- Periodic evaluation and logging
- Model checkpointing
- Early stopping detection

### Phase 5: Completion
- Final model save
- Training complete message