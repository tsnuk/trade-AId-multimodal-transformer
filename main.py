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

# Import our modular components
from compatibility_layer import (
    initialize_compatibility_layer, get_modality_parameters,
    get_system_configuration
)
from data_utils import (
    load_file_data, range_numeric_data, report_non_numeric_error,
    bin_numeric_data, numerical_representation, create_train_val_datasets,
    write_initial_run_details
)
from file_cache import load_file_data_cached, print_cache_stats, cleanup_cache
from model import MultimodalTransformer
from training_utils import get_batch, estimate_loss
from progress_utils import show_progress_bar, show_stage_progress, finish_progress_line

"""# Data loading & processing"""

# Initialize configuration system (supports both YAML and programmatic setup)
# The system supports two configuration methods:
# - YAML Configuration: Uses input_schemas.yaml + config.yaml files (recommended for most users)
# - Programmatic Configuration: Uses Python variables in config.py (advanced users, automation)
print("Initializing Trade-AId Multimodal Transformer...")
print("Initializing configuration system...")
config_mode = initialize_compatibility_layer(globals())
print(f"Configuration: {'YAML mode detected' if config_mode == 'modern' else 'Programmatic mode detected'}")

# Get configuration parameters through compatibility layer
system_config = get_system_configuration()
modality_params_list = get_modality_parameters()

if not modality_params_list:
    raise ValueError("No modalities configured. Please check your configuration files.")

print(f"Modalities: Loaded {len(modality_params_list)} configurations")
print()  # Spacing before data loading

# Extract individual configuration variables from system_config
# This ensures compatibility with both YAML and programmatic configurations
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

# Data Preparation:
# - Load raw data from files
# - Process the data (if specified)
# - Create a vocabulary of unique elements and convert it into a numerical representation
# - Split the data into training and validation sets

all_modality_data = []  # For each modality, will contain a list of raw data elements, or of processed elements (if specified and if numeric)
all_file_info = []  # For each modality, will contain a list of the loaded file information: [file1_name, data1_length, file2_name, data2_length, ...]
all_modality_params = []  # For each modality, will contain a list of processing parameters

modality_num = 0
is_percents = False

print(f"Data Loading: Processing {len(modality_params_list)} modalities...")

for i, modality_params in enumerate(modality_params_list):
    # Extract parameters from modality configuration
    # Format: [path, column_number, has_header, convert_to_percents, num_whole_digits, decimal_places, num_bins, rand_size, cross_attend, modality_name]

    convert_to_percents = modality_params[3] if len(modality_params) > 3 else False
    num_whole_digits = modality_params[4] if len(modality_params) > 4 else None
    decimal_places = modality_params[5] if len(modality_params) > 5 else None
    num_bins = modality_params[6] if len(modality_params) > 6 else None
    rand_size = modality_params[7] if len(modality_params) > 7 else None
    cross_attend = modality_params[8] if len(modality_params) > 8 else False
    modality_name = modality_params[9] if len(modality_params) > 9 else ''

    if convert_to_percents:
        is_percents = True

    # Show progress for current modality
    show_stage_progress(i + 1, len(modality_params_list), f"Loading '{modality_name}'")

    modality_num += 1

    # Load data using complete modality parameters (with caching for performance)
    this_modality_data, this_file_info = load_file_data_cached(modality_params)

    # Range numeric data: scale values and set decimal places
    if num_whole_digits is not None or decimal_places is not None:
        # Check if the loaded data is numeric before processing
        data_is_numeric = all(isinstance(item, numbers.Number) for item in this_modality_data)
        if data_is_numeric:
            print(f"  Processing: Applying ranging/decimal places to '{modality_name}'")
            this_modality_data = range_numeric_data(this_modality_data, num_whole_digits, decimal_places)
        else:
            # Find and report the non-numeric element
            print(f"  Warning: Ranging/decimal places specified for '{modality_name}' but data is not numeric")
            report_non_numeric_error(this_modality_data, this_file_info, modality_num)

    # Bin numeric data
    if num_bins is not None:
        outlier_percentile = 0.1 # Percentage of extreme values (outliers) to be excluded from bin range calculation
        exponent = 2.2 # Controls how bin ranges are distributed
        print(f"  Processing: Applying binning to '{modality_name}'")
        this_modality_data = bin_numeric_data(this_modality_data, num_bins, outlier_percentile, exponent)

    all_modality_data.append(this_modality_data)
    all_file_info.append(this_file_info)
    all_modality_params.append(modality_params)

# Finish the data loading progress
finish_progress_line()


print("Data Loading: Complete")
num_modalities = len(all_modality_data)

# Check for equal modality lengths
if num_modalities > 1:
    first_modality_length = len(all_modality_data[0])
    for i in range(1, num_modalities):
        if len(all_modality_data[i]) != first_modality_length:
            raise ValueError(
                f"Modality {i+1} has a different data length ({len(all_modality_data[i])}) "
                f"than the first modality ({first_modality_length}). "
                "All modalities must have the same length for proper training."
            )

print("Vocabulary Building: Creating numerical representations...")

all_vocabularies = []
all_numeric_reps = []

for m in range(num_modalities):
  # Access modality name using the attribute from the ModalityConfig instance
  this_modality_name = all_modality_params[m][9] if all_modality_params[m][9] is not None else f"Modality {m+1}"

  # Show progress for vocabulary building
  show_progress_bar(m, num_modalities, f"Building vocabularies")

  # Generate a vocabulary and numerical representation
  this_numeric_rep, this_vocabulary = numerical_representation(all_modality_data[m])

  all_numeric_reps.append(this_numeric_rep)
  all_vocabularies.append(this_vocabulary)

  # Show vocabulary info (on a new line after progress bar)
  print(f"  Modality {m+1} '{this_modality_name}': Vocabulary size {len(this_vocabulary)}")
  if len(this_vocabulary) <= 20:
    print(f"    Vocabulary: {this_vocabulary}")
  else:
    print(f"    Vocabulary (first 10): {this_vocabulary[:10]}")

# Complete the vocabulary building progress
show_progress_bar(num_modalities, num_modalities, f"Building vocabularies")

print("="*60)

# Get file lengths for splitting
file_lengths = []
# Use the file lengths from the *first* modality for splitting consistency across all modalities
if all_file_info and len(all_file_info) > 0:
  for f_idx in range(1, len(all_file_info[0]), 2):
    file_lengths.append(all_file_info[0][f_idx])
else:
  # Fallback if no file info is available
  file_lengths = [len(all_modality_data[0])]

print("Dataset Splitting: Creating train/validation sets...")

all_train_sets = []
all_val_sets = []

for i in range(num_modalities):
  # Show progress for dataset splitting
  modality_name = all_modality_params[i][9] if all_modality_params[i][9] else f"Modality {i+1}"
  show_progress_bar(i, num_modalities, f"Splitting datasets")

  # Use the file_lengths derived from the first modality for splitting all modalities
  this_train_set, this_val_set = create_train_val_datasets(all_numeric_reps[i], system_config['validation_size'], system_config['num_validation_files'], file_lengths)
  all_train_sets.append(this_train_set)
  all_val_sets.append(this_val_set)

# Complete the dataset splitting progress
show_progress_bar(num_modalities, num_modalities, f"Splitting datasets")

# Print validation method information
if system_config['num_validation_files'] > 0:
    print("Validation: File-based splitting")
    print("  Validation files:")
    # For the validation set we need to go backwards, so start from the second to last element (index len(all_file_info[0]) - 2) and step backwards by 2
    val_files_counter = 0
    for j in range(len(all_file_info[0]) - 2, -1, -2):
        this_file_name = all_file_info[0][j]
        print(f"    - {this_file_name}")
        val_files_counter += 1
        if val_files_counter >= system_config['num_validation_files']:
            break
else:
    print(f"Validation: Percentage-based splitting ({system_config['validation_size']*100:.1f}% validation)")

# Show file caching statistics
print_cache_stats()

# Clean up cache to free memory (files are no longer needed)
cleanup_cache()

print("Data Preparation: Complete")
print()  # Spacing before model section

"""# Building the transformer"""

# Set global variables in training_utils
import training_utils
training_utils.all_full_datasets = all_numeric_reps  # Pass full datasets for proper index generation
training_utils.all_train_sets = all_train_sets
training_utils.all_val_sets = all_val_sets
training_utils.all_vocabularies = all_vocabularies
training_utils.all_modality_params = all_modality_params
training_utils.all_file_info = all_file_info
training_utils.file_lengths = file_lengths
training_utils.num_modalities = num_modalities
training_utils.is_percents = is_percents

# Create vocab sizes list for model initialization
all_vocab_sizes = [len(vocab) for vocab in all_vocabularies]

# Model parameter count calculation
model_params = n_embd * sum(all_vocab_sizes) + n_embd * block_size + n_layer * (
    num_modalities * (3 * n_embd * n_embd + n_embd * 4 * n_embd + 4 * n_embd + n_embd)
    + sum(vocab_size * n_embd for vocab_size in all_vocab_sizes)
)

print("Model Configuration:")
print(f"  Modalities: {num_modalities}")
print(f"  Vocabulary sizes: {all_vocab_sizes}")
print(f"  Parameters: {model_params/1e6:.1f}M")
print()  # Spacing before training section

if create_new_model == 1:
    print("Model: Creating new transformer...")
    # Pass the list of vocab sizes and all_modality_params to the model constructor
    m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    print("Model: Created successfully")
else:
    print(f"Model: Loading from {model_file_name}...")
    # Pass the list of vocab sizes and all_modality_params when instantiating the model for loading
    m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
    try:
        m.load_state_dict(torch.load(model_file_name, weights_only=True))
        print("Model: Loaded successfully")
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Optimizer: Created with loaded parameters")
    except FileNotFoundError:
        print(f"Model: File not found, creating new model instead")
        # Pass the list of vocab sizes and all_modality_params to the model constructor
        m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Model: Created successfully")
    except Exception as e:
        print(f"Model: Loading failed ({e}), creating new model")
        # Pass the list of vocab sizes and all_modality_params to the model constructor
        m = MultimodalTransformer(num_modalities, all_vocab_sizes, all_modality_params).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        print("Model: Created successfully")

# Set model in training_utils
training_utils.m = m

# Prepare data for output file
hyperparams = {
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "batch_size": batch_size,
    "dropout": dropout,
    "learning_rate": learning_rate
}

# Extract vocab sizes and data lengths for data_info summary
modality_vocab_sizes_summary = ", ".join([f"Modality {i+1}={len(all_vocabularies[i])}" for i in range(num_modalities)])
modality_data_lengths_summary = ", ".join([f"Modality {i+1}={len(all_modality_data[i])}" for i in range(num_modalities)])


# 1. Hyperparameters dictionary
# (already created above)

# 2. Run Statistics dictionary
run_stats = {
    "Model parameter size (M)": round(model_params / 1e6, 1)
}

# 3. Data Information dictionary
# Assuming train/val sizes are the same for all modalities
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

# 4. Modality Configurations list of dictionaries
modality_configs = []
for i in range(num_modalities):
    modality_params = all_modality_params[i] # This is a ModalityConfig instance
    modality_file_info = all_file_info[i]

    # Extract the first file name as the source
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

# Write initial run details to output file
output_file_path = project_file_path + 'output/' + output_file_name
if output_file_name != '':
    write_initial_run_details(output_file_path, hyperparams, data_info, modality_configs, run_stats)
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write("\\n\\n--- Evaluation Results ---\\n") # Add the header

"""# Running the transformer"""

# Training loop
print()  # Spacing before training
print(f"Training: Starting {max_iters} iterations on {device}")

# Early stopping variables
best_val_loss = float('inf')
patience = 1000  # Number of evaluations to wait for improvement
no_improvement_count = 0

for iter in range(max_iters): # the loop iterates for a maximum number of iterations (max_iters)
                              # it periodically estimates the loss and prints it
                              # it also generates text samples using the model's generate method

    # Every once in a while evaluate the loss on train and val sets
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if iter % 100 == 0 : print(f'Training: Iteration {iter}/{max_iters}')
    if iter % eval_interval == 0 or iter == max_iters - 1:
        # Pass the warning tracking list to estimate_loss
        print(f"Evaluation: Step {iter}...")
        losses = estimate_loss() # Uses eval_iters from global scope
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # Check if losses are valid before printing
        if not torch.isnan(torch.tensor([losses['train'], losses['val']])).any():
             print(f"{'='*70}")
             print(f"🎯 RESULTS: Step {iter} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | {current_time}")
             print(f"{'='*70}")
             # write to file
             if output_file_name != '':
               with open(output_file_path, 'a', encoding='utf-8') as f:
                   f.write(f"Step {iter} Summary: Training Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f} | Time: {current_time}\\n\\n")
        else:
             print(f"Warning: Step {iter} losses are NaN, skipping save | {current_time}")

        # Early stopping logic
        # if the validation loss doesn't improve for a certain number of iterations (patience), the training process is stopped
        # Only apply early stopping if validation loss is a valid number
        if not torch.isnan(torch.tensor(losses['val'])).any(): # Use .any() for tensor check
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Training: Early stopping (no improvement for {patience} evaluations)")
                break

    # Save model periodically if save_model is enabled
    if save_model == 1 and (iter % eval_interval == 0 or iter == max_iters - 1):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'Saving: Model checkpoint | {current_time}')
        # When saving, save the state dict of the MultimodalTransformer model
        # Need to ensure model_file_name includes the full path if project_file_path is used
        # model_file_name is loaded as a full path in S3fmsYL-7lVQ
        torch.save(m.state_dict(), model_file_name)
        print(f"Saved: {round(os.path.getsize(model_file_name)/1024**2,2)} MB")


    # Training steps
    # get_batch returns lists of tensors: [xb_mod1, xb_mod2, ...], [yb_mod1, yb_mod2, ...]
    # get_batch needs access to all_train_sets, all_val_sets, device, block_size, batch_size, randomness_size
    # randomness_size is in all_modality_params, which is accessible globally
    # all_modality_params is a list of ModalityConfig instances, need to pass this to get_batch
    xb_list, yb_list = get_batch('train', 1) # Pass 1 for is_training flag


    # Pass lists of tensors to the multimodal model
    # m is the model instance
    logits_list, losses_list = m(xb_list, yb_list)


    # Calculate total loss by summing modality losses
    # Ensure losses_list is not None and contains tensors before summing
    if losses_list and all(l is not None for l in losses_list):
        total_loss = sum(losses_list)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward() # Backpropagate the combined loss
        optimizer.step()
    else:
        # Handle cases where losses might not be calculated (e.g., if targets were None, though get_batch for 'train' should provide them)
        print("Warning: Training step losses not calculated, skipping backpropagation")

    '''
    line 1: zero_grad() clears the gradients accumulated from the previous training step. By default, PyTorch accumulates gradients, so we need to clear them before computing the gradients for the current step.
    line 2: this line calls the backward() method on the loss tensor, which triggers the backpropagation algorithm to calculate the gradients of the loss with respect to all the model parameters.
    line 3: this line updates the model's parameters using the gradients calculated in the previous step.
    line 4: this line defines a variable `losses` which will store the estimated losses on the training and validation sets every `eval_interval` steps.
    line 5: this line updates the model's parameters using the calculated gradients
            optimizer.step()
            the optimizer (AdamW) takes a step towards minimizing the loss by adjusting the parameters in the direction indicated by the gradients
    '''

print("Training: Completed successfully")

# Final model save
if save_model == 1:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'Final Save: Model checkpoint | {current_time}')
    torch.save(m.state_dict(), model_file_name)
    print(f"Final Save: {round(os.path.getsize(model_file_name)/1024**2,2)} MB complete")

"""#End New Code"""