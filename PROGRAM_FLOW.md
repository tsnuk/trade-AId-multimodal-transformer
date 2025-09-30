# MULTIMODAL TRANSFORMER TRAINING FLOW

## Complete Program Execution Flow

```
START: main.py execution
│
|- 1. CONFIGURATION LOADING (main.py:30-60)
|   |- config_manager.py -> load_config()
|   |   |- Loads config.yaml -> system settings
|   |   '- Loads input_schemas.yaml -> modality configurations
|   |- schema.py -> SchemaManager.load_from_yaml()
|   |   |- Validates modality configurations
|   |   '- Creates InputSchema objects
|   '- Sets global config variables
│
|- 2. DATA LOADING & PROCESSING (main.py:61-235)
|   |- For each modality:
|   |   |- data_utils.py -> load_and_process_data()
│   │   │   ├─ Loads CSV/TXT files from path
│   │   │   ├─ Extracts specified column
│   │   │   └─ Returns raw numeric data + file info
│   │   ├─ processing_pipeline.py → ProcessingPipeline.process()
│   │   │   ├─ Applies processing steps sequentially:
│   │   │   │   ├─ convert_to_percent_changes()
│   │   │   │   ├─ range_numeric_data()
│   │   │   │   └─ bin_numeric_data()
│   │   │   └─ processing_registry.py → validates functions
│   │   └─ Creates vocabulary from processed data
│   └─ 📊 VOCABULARY BUILDING output
│
├─ 3️⃣ DATASET SPLITTING (main.py:245-280)
│   ├─ data_utils.py → create_train_val_datasets()
│   │   ├─ If num_validation_files > 0: File-based split
│   │   └─ Else: Percentage-based split using validation_size
│   └─ 🗂️ DATASET SPLITTING output
│
├─ 4️⃣ MODEL INITIALIZATION (main.py:282-320)
│   ├─ model.py → MultimodalTransformer()
│   │   ├─ Creates embedding layers for each modality
│   │   ├─ Initializes MultimodalBlock layers
│   │   │   ├─ MultiHeadAttention (self-attention)
│   │   │   ├─ CrossAttention (cross-modal attention)
│   │   │   └─ FeedForward layers
│   │   └─ MultimodalPreBlock/PostBlock
│   ├─ If create_new_model=0: Load existing model weights
│   └─ Move model to device (CPU/CUDA)
│
├─ 5️⃣ TRAINING SETUP (main.py:321-390)
│   ├─ Creates PyTorch optimizer (AdamW)
│   ├─ Sets up training log file
│   ├─ data_utils.py → write_training_log_header()
│   │   └─ Writes configuration summary to log
│   └─ 🚀 TRAINING STARTUP output
│
├─ 6️⃣ MAIN TRAINING LOOP (main.py:391-520)
│   │
│   ├─ For each iteration (0 to max_iters):
│   │   │
│   │   ├─ 🔄 TRAINING STEP
│   │   │   ├─ training_utils.py → train_step()
│   │   │   │   ├─ Gets random training batches
│   │   │   │   ├─ model.forward() → MultimodalTransformer
│   │   │   │   │   ├─ Embedding lookup for each modality
│   │   │   │   │   ├─ Processes through transformer blocks:
│   │   │   │   │   │   ├─ Self-attention (all modalities)
│   │   │   │   │   │   ├─ Cross-attention (if enabled)
│   │   │   │   │   │   └─ FeedForward
│   │   │   │   │   └─ Output predictions
│   │   │   │   ├─ Calculates loss (CrossEntropyLoss)
│   │   │   │   ├─ Backpropagation
│   │   │   │   └─ Optimizer step
│   │   │   └─ Returns training loss
│   │   │
│   │   ├─ 📊 EVALUATION (every eval_interval steps)
│   │   │   ├─ training_utils.py → calculate_evaluation_metrics()
│   │   │   │   ├─ Runs model on validation batches
│   │   │   │   ├─ Calculates validation loss
│   │   │   │   ├─ Calculates directional accuracy
│   │   │   │   └─ Returns metrics dictionary
│   │   │   ├─ 🎯 LOSS METRICS output (console)
│   │   │   ├─ 📈 STEP log entry (training_log.txt)
│   │   │   └─ Model saving (if save_model=1)
│   │   │
│   │   └─ Early stopping check (if validation improves)
│   │
│   └─ Loop continues until max_iters reached
│
├─ 7️⃣ TRAINING COMPLETION (main.py:521-560)
│   ├─ Final model save (if save_model=1)
│   ├─ Final evaluation metrics
│   ├─ Cleanup (file_cache.py → cleanup_cache())
│   └─ 🏁 TRAINING COMPLETE output
│
└─ 🔚 END: Program termination
```

## 📁 Key Files and Their Roles

| File | Primary Responsibility |
|------|----------------------|
| **main.py** | Main execution orchestrator |
| **config_manager.py** | Configuration loading/validation |
| **schema.py** | YAML schema management |
| **data_utils.py** | Data loading, processing, splitting |
| **processing_pipeline.py** | Data processing coordination |
| **processing_registry.py** | Function validation/resolution |
| **model.py** | Neural network architecture |
| **training_utils.py** | Training/evaluation operations |
| **file_cache.py** | File caching and cleanup |

## 🗂️ Configuration Files

| File | Purpose |
|------|---------|
| **config.yaml** | System settings & hyperparameters |
| **input_schemas.yaml** | Modality definitions & processing |

## 📊 Output Files

| File | Content |
|------|---------|
| **training_log.txt** | Detailed training log |
| **TransformerModel.pth** | Saved model weights |

## 🔄 Key Processing Phases

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
- Cleanup and termination