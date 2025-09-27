"""main.py

Main execution script for the multimodal transformer training system.

Orchestrates the entire training process:
- Data loading and preprocessing
- Model creation and configuration
- Training loop execution
- Evaluation and metrics tracking

"""

import torch
import numbers
import os
from datetime import datetime

from compatibility_layer import (
    initialize_compatibility_layer, get_modality_parameters,
    get_system_configuration, is_modern_mode
)
from data_utils import (
    load_file_data, range_numeric_data, report_non_numeric_error,
    bin_numeric_data, numerical_representation, create_train_val_datasets,
    write_initial_run_details
)
from file_cache import load_file_data_cached, cleanup_cache
from model import MultimodalTransformer
from training_utils import get_batch, estimate_loss
print("🚀 TRADE-AID MULTIMODAL TRANSFORMER")
print("═" * 45)
print("Initializing configuration system...")
config_mode = initialize_compatibility_layer(globals())
print(f"Configuration: {'YAML mode detected' if config_mode == 'modern' else 'Programmatic mode detected'}")
print()

system_config = get_system_configuration()
modality_params_list = get_modality_parameters()

if not modality_params_list:
    raise ValueError("No modalities configured. Please check your configuration files.")

print(f"Modalities: Loaded {len(modality_params_list)} configurations")
print()

batch_size = system_config['batch_size']
block_size = system_config['block_size']
max_iters = system_config['max_iters']
eval_interval = system_config['eval_interval']
eval_iters = system_config['eval_iters']
learning_rate = system_config['learning_rate']
device = system_config['device']
n_embd = system_config['n_embd']
n_head = system_config['n_head']
n_layer = system_config['n_layer']
dropout = system_config['dropout']
validation_size = system_config['validation_size']
num_validation_files = system_config['num_validation_files']
create_new_model = system_config['create_new_model']
save_model = system_config['save_model']
model_file_name = system_config['model_file_name']
project_file_path = system_config['project_file_path']
output_file_name = system_config['output_file_name']

all_modality_data = []
all_file_info = []
all_modality_params = []
all_raw_vocab_sizes = []

modality_num = 0
is_percents = False

print(f"Data Loading: Processing {len(modality_params_list)} modalities...")

for i, modality_params in enumerate(modality_params_list):

    convert_to_percents = modality_params[3] if len(modality_params) > 3 else False
    num_whole_digits = modality_params[4] if len(modality_params) > 4 else None
    decimal_places = modality_params[5] if len(modality_params) > 5 else None
    num_bins = modality_params[6] if len(modality_params) > 6 else None
    rand_size = modality_params[7] if len(modality_params) > 7 else None
    cross_attend = modality_params[8] if len(modality_params) > 8 else False
    modality_name = modality_params[9] if len(modality_params) > 9 else ''

    if convert_to_percents:
        is_percents = True

    print(f"  Loading modality {i + 1}: '{modality_name}'")

    modality_num += 1

    this_modality_data, this_file_info = load_file_data_cached(modality_params)

    raw_vocabulary_size = len(set(this_modality_data))

    processing_applied = False

    if convert_to_percents:
        print(f"    Processing: Converting to percentages")
        processing_applied = True

    if num_whole_digits is not None or decimal_places is not None:
        data_is_numeric = all(isinstance(item, numbers.Number) for item in this_modality_data)
        if data_is_numeric:
            range_details = f"{num_whole_digits} digits" if num_whole_digits else ""
            decimal_details = f"{decimal_places} decimals" if decimal_places else ""
            details = ", ".join(filter(None, [range_details, decimal_details]))
            print(f"    Processing: Ranging ({details})")
            this_modality_data = range_numeric_data(this_modality_data, num_whole_digits, decimal_places)
            processing_applied = True
        else:
            print(f"    Warning: Ranging/decimal places specified but data is not numeric")
            report_non_numeric_error(this_modality_data, this_file_info, modality_num)

    if num_bins is not None:
        outlier_percentile = 0.1
        exponent = 2.2
        print(f"    Processing: Binning ({num_bins} positive, {num_bins} negative, 1 zero bins)")
        this_modality_data = bin_numeric_data(this_modality_data, num_bins, outlier_percentile, exponent)
        processing_applied = True

    if not processing_applied:
        if is_modern_mode():
            from compatibility_layer import compatibility_layer
            modality_metadata = compatibility_layer.get_modality_metadata(modality_num - 1)  # 0-based index
            if modality_metadata.get('processing_steps_count', 0) > 0:
                if compatibility_layer.config_manager:
                    schemas = compatibility_layer.config_manager.schema_manager.schemas
                    if modality_num - 1 < len(schemas):
                        schema = schemas[modality_num - 1]
                        external_functions = [step.function for step in schema.processing_steps if step.enabled]
                        if external_functions:
                            external_names = ', '.join(external_functions)
                            print(f"    Processing: External functions ({external_names})")
                        else:
                            print(f"    Processing: No processing specified")
                    else:
                        print(f"    Processing: No processing specified")
                else:
                    print(f"    Processing: No processing specified")
            else:
                print(f"    Processing: No processing specified")
        else:
            print(f"    Processing: No processing specified")

    all_modality_data.append(this_modality_data)
    all_file_info.append(this_file_info)
    all_modality_params.append(modality_params)
    all_raw_vocab_sizes.append(raw_vocabulary_size)

    data_length = len(this_modality_data)
    file_count = len(this_file_info) // 2 if this_file_info else 0
    print(f"  Summary: {data_length:,} data points ({file_count} files loaded)")
    if i < len(modality_params_list) - 1:  # Add separator except after last modality
        print()

print("Data Loading and Processing: Complete")
print()
num_modalities = len(all_modality_data)
if num_modalities > 1:
    first_modality_length = len(all_modality_data[0])
    for i in range(1, num_modalities):
        if len(all_modality_data[i]) != first_modality_length:
            raise ValueError(
                f"Modality {i+1} has a different data length ({len(all_modality_data[i])}) "
                f"than the first modality ({first_modality_length}). "
                "All modalities must have the same length for proper training."
            )

print("\n📊 VOCABULARY BUILDING")

all_vocabularies = []
all_numeric_reps = []

for m in range(num_modalities):
  this_modality_name = all_modality_params[m][9] if all_modality_params[m][9] is not None else f"Modality {m+1}"
  this_numeric_rep, this_vocabulary = numerical_representation(all_modality_data[m])

  all_numeric_reps.append(this_numeric_rep)
  all_vocabularies.append(this_vocabulary)

  raw_vocab_size = all_raw_vocab_sizes[m]

  processing_applied = []
  modality_params = all_modality_params[m]

  if is_modern_mode():
    from compatibility_layer import compatibility_layer
    if compatibility_layer.config_manager:
      schemas = compatibility_layer.config_manager.schema_manager.schemas
      if m < len(schemas):
        schema = schemas[m]
        for step in schema.processing_steps:
          if step.enabled:
            if step.function == 'convert_to_percent_changes':
              processing_applied.append("percentages")
            elif step.function == 'range_numeric_data':
              processing_applied.append("ranging")
            elif step.function == 'bin_numeric_data':
              processing_applied.append("binning")
            else:
              processing_applied.append(step.function)
  else:
    if len(modality_params) > 3 and modality_params[3]:
      processing_applied.append("percentages")
    if len(modality_params) > 4 and modality_params[4] is not None:
      processing_applied.append("ranging")
    if len(modality_params) > 6 and modality_params[6] is not None:
      processing_applied.append("binning")

  processing_text = f" ({'+'.join(processing_applied)})" if processing_applied else ""

  print(f"  ▪ {this_modality_name:<30} {raw_vocab_size:,} → {len(this_vocabulary):,}  {processing_text.strip('() ') if processing_text.strip() else 'no processing'}")

  if len(this_vocabulary) <= 20:
    print(f"    Vocabulary: {this_vocabulary}")
  else:
    truncated_vocab = this_vocabulary[:10] + ['...']
    vocab_str = str(truncated_vocab).replace("'...'", "...")
    print(f"    Vocabulary: {vocab_str}")

file_lengths = []
if all_file_info and len(all_file_info) > 0:
  for f_idx in range(1, len(all_file_info[0]), 2):
    file_lengths.append(all_file_info[0][f_idx])
else:
  file_lengths = [len(all_modality_data[0])]

print()
print("Dataset Splitting: Creating train/validation sets...")

if system_config['num_validation_files'] > 0:
    print("Method: File-based splitting")
    print("  Validation files:")
    val_files_counter = 0
    for j in range(len(all_file_info[0]) - 2, -1, -2):
        this_file_name = all_file_info[0][j]
        print(f"    - {this_file_name}")
        val_files_counter += 1
        if val_files_counter >= system_config['num_validation_files']:
            break
else:
    print(f"Method: Percentage-based splitting ({system_config['validation_size']*100:.1f}% validation)")

print("\n🗂️ DATASET SPLITTING")
all_train_sets = []
all_val_sets = []

for i in range(num_modalities):
  modality_name = all_modality_params[i][9] if all_modality_params[i][9] else f"Modality {i+1}"
  rand_size = all_modality_params[i][7] if len(all_modality_params[i]) > 7 and all_modality_params[i][7] is not None else None
  rand_text = f" | Randomness: {rand_size}" if rand_size is not None else ""

  cross_attention = all_modality_params[i][8] if len(all_modality_params[i]) > 8 and all_modality_params[i][8] is not None else True
  cross_text = " | Cross-attention: ON" if cross_attention else " | Cross-attention: OFF"

  this_train_set, this_val_set = create_train_val_datasets(all_numeric_reps[i], system_config['validation_size'], system_config['num_validation_files'], file_lengths)
  all_train_sets.append(this_train_set)
  all_val_sets.append(this_val_set)

  print(f"  ▪ {modality_name:<30}Train {len(this_train_set):,} | Val {len(this_val_set):,}{rand_text}{cross_text}")

cleanup_cache()

print()
print("Data Preparation: Complete")
print()

# Set global variables in training_utils
import training_utils
training_utils.all_full_datasets = all_numeric_reps
training_utils.all_train_sets = all_train_sets
training_utils.all_val_sets = all_val_sets
training_utils.all_vocabularies = all_vocabularies
training_utils.all_modality_params = all_modality_params
training_utils.all_file_info = all_file_info
training_utils.file_lengths = file_lengths
training_utils.num_modalities = num_modalities
training_utils.is_percents = is_percents

all_vocab_sizes = [len(vocab) for vocab in all_vocabularies]

# Calculate model parameters following standard conventions
# 1. Embeddings
token_embeddings = sum(vocab_size * n_embd for vocab_size in all_vocab_sizes)
positional_embeddings = block_size * n_embd

# 2. Per transformer layer parameters
def count_cross_attention_params():
    total = 0
    for i, modality_params in enumerate(all_modality_params):
        if modality_params[8]:  # cross_attention enabled
            num_other_modalities = num_modalities - 1
            # Cross-attention has similar structure to self-attention but with multiple KV sources
            total += num_other_modalities * (2 * (n_embd * (n_head * (n_embd // n_head) // 2) + (n_embd // n_head) // 2 * (n_embd // n_head))) + n_embd * (n_embd // 2) + (n_embd // 2) * n_embd
            total += n_embd  # LayerNorm for cross-attention
    return total

# Per layer: self-attention + feedforward + layer norms for each modality
per_layer_params = 0
for i in range(num_modalities):
    # Self-attention: each head has 3 projections (Q,K,V) + output projection
    head_size = n_embd // n_head
    # Each head: 3 * (n_embd * head_size//2 + head_size//2 * head_size)
    attention_params = n_head * 3 * (n_embd * (head_size // 2) + (head_size // 2) * head_size)
    # Output projection: head_size*n_head -> n_embd//2 -> n_embd
    attention_params += (head_size * n_head) * (n_embd // 2) + (n_embd // 2) * n_embd

    # FeedForward: n_embd -> 4*n_embd -> n_embd
    feedforward_params = n_embd * (4 * n_embd) + (4 * n_embd) * n_embd

    # LayerNorms: ln1 and ln2 (weight only, no bias by default)
    layernorm_params = 2 * n_embd

    per_layer_params += attention_params + feedforward_params + layernorm_params

# Cross-attention parameters
cross_attention_params = count_cross_attention_params()

# 3. Output head parameters
output_params = 0
for i in range(num_modalities):
    vocab_size = all_vocab_sizes[i]
    # LayerNorm + two linear layers: n_embd -> vocab_size//2 -> vocab_size
    output_params += n_embd + n_embd * (vocab_size // 2) + (vocab_size // 2) * vocab_size

model_params = token_embeddings + positional_embeddings + n_layer * (per_layer_params + cross_attention_params) + output_params

print("="*60)
print("MODEL CREATION & TRAINING")
print("="*60)
print()
print("Model Configuration:")
print(f"  Modalities: {num_modalities}")
print(f"  Vocabulary sizes: {all_vocab_sizes}")
print(f"  Parameters: {model_params/1e6:.1f}M")
print()

if create_new_model == 1:
    print("Model: Creating new transformer...")
    m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    print("Model: Created successfully")
else:
    print(f"Model: Loading from {model_file_name}...")
    m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
    try:
        m.load_state_dict(torch.load(model_file_name, weights_only=True))
        print("Model: Loaded successfully")
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Optimizer: Created with loaded parameters")
    except FileNotFoundError:
        print(f"Model: File not found, creating new model instead")
        m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Model: Created successfully")
    except Exception as e:
        print(f"Model: Loading failed ({e}), creating new model")
        m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Model: Created successfully")

training_utils.m = m

hyperparams = {
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "batch_size": batch_size,
    "dropout": dropout,
    "learning_rate": learning_rate
}

modality_vocab_sizes_summary = ", ".join([f"Modality {i+1}={len(all_vocabularies[i])}" for i in range(num_modalities)])
modality_data_lengths_summary = ", ".join([f"Modality {i+1}={len(all_modality_data[i])}" for i in range(num_modalities)])


run_stats = {
    "Model parameter size (M)": round(model_params / 1e6, 1)
}

train_size = len(all_train_sets[0])
val_size_actual = len(all_val_sets[0])
split_method = f"validation_size={validation_size}" if num_validation_files == 0 else f"num_validation_files={num_validation_files}"

data_info = {
    "Number of modalities": num_modalities,
    "Train set size": train_size,
    "Val set size": val_size_actual,
    "Split method": split_method,
    "Modality vocabulary sizes": modality_vocab_sizes_summary,
    "Modality data lengths": modality_data_lengths_summary
}

modality_configs = []
for i in range(num_modalities):
    modality_params = all_modality_params[i]
    modality_file_info = all_file_info[i]
    source_file = modality_file_info[0] if modality_file_info else "Unknown"

    config_dict = {
        "Source": source_file,
        "Modality Name": modality_params[6] if modality_params[6] else f"Modality {i+1}",
        "Num Whole Digits": modality_params[0],
        "Decimal Places": modality_params[1],
        "Rand Size": modality_params[2],
        "Cross-Attend": modality_params[3],
        "Convert to Percents": modality_params[4],
        "Num Bins": modality_params[5],
        "Original Col Num": "N/A (not in processed params)",
        "Original Has Header": "N/A (not in processed params)"
    }
    modality_configs.append(config_dict)

output_file_path = project_file_path + 'output/' + output_file_name
if output_file_name != '':
    write_initial_run_details(output_file_path, hyperparams, data_info, modality_configs, run_stats)
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write("\\n\\n--- Evaluation Results ---\\n")

print()
print(f"🔄 TRAINING PROGRESS")
print(f"  ▪ Iterations: {max_iters}")
print(f"  ▪ Device: {device}")
print("  ▪ Note: Intensive computation ahead")
print()

best_val_loss = float('inf')
patience = 1000
no_improvement_count = 0

for iter in range(max_iters):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if iter % 100 == 0 : print(f'Training: Iteration {iter}/{max_iters}')
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(iter, max_iters)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if not torch.isnan(torch.tensor([losses['train'], losses['val']])).any():
             print(f"\n🎯 LOSS METRICS: Step {iter}/{max_iters} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | Time: {current_time}")
             print("─" * 80)
             if output_file_name != '':
               with open(output_file_path, 'a', encoding='utf-8') as f:
                   f.write(f"Step {iter} Summary: Training Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f} | Time: {current_time}\\n\\n")
        else:
             print(f"Warning: Step {iter} losses are NaN, skipping save | {current_time}")

        if not torch.isnan(torch.tensor(losses['val'])).any():
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Training: Early stopping (no improvement for {patience} evaluations)")
                break

    if save_model == 1 and (iter % eval_interval == 0 or iter == max_iters - 1):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        torch.save(m.state_dict(), model_file_name)
        print()
        print(f'Saved: Model checkpoint ({round(os.path.getsize(model_file_name)/1024**2,2)} MB) | {current_time}')
        print()


    xb_list, yb_list = get_batch('train', 1)
    logits_list, losses_list = m(xb_list, yb_list)


    if losses_list and all(l is not None for l in losses_list):
        total_loss = sum(losses_list)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
    else:
        print("Warning: Training step losses not calculated, skipping backpropagation")


print("\\n✅ TRAINING COMPLETED SUCCESSFULLY")

if save_model == 1:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'Final Save: Model checkpoint | {current_time}')
    torch.save(m.state_dict(), model_file_name)
    print(f"Final Save: {round(os.path.getsize(model_file_name)/1024**2,2)} MB complete")

