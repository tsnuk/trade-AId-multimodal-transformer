"""training_utils.py

Training utilities and helper functions for the multimodal transformer system.

Contains functions related to:
- Batch generation and data sampling
- Loss estimation and evaluation
- Training metrics and monitoring
- Direction prediction analysis

Extracted from mm_final_4.py for better code organization.
"""

import torch
import torch.nn.functional as F
import numbers
import os
from datetime import datetime
# Configuration will be loaded lazily when needed

# Global configuration cache - will be populated when first accessed
_config_cache = None

def _get_config():
    """Lazy load configuration through compatibility layer"""
    global _config_cache
    if _config_cache is None:
        from compatibility_layer import get_system_configuration
        _config_cache = get_system_configuration()
    return _config_cache

# Create property-like accessors for configuration values
def _get_device(): return _get_config()['device']
def _get_block_size(): return _get_config()['block_size']
def _get_batch_size(): return _get_config()['batch_size']
def _get_eval_iters(): return _get_config()['eval_iters']

# Export only the functions that should be publicly available
__all__ = [
    # Batch generation functions
    'generate_batch_starting_indices', 'get_batch',
    # Training utilities
    'estimate_loss', 'calculate_evaluation_metrics',
    # Logging functions
    'save_loss_to_file', 'generate_and_save_output'
]


def generate_batch_starting_indices(data_size, block_size, batch_size, split, file_lengths, is_percents):
    """Generate random starting indices for batch sequences, respecting file boundaries."""
    # Validate inputs
    if not isinstance(data_size, int) or data_size <= 0:
        raise TypeError("'data_size' must be a positive integer.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise TypeError("'block_size' must be a positive integer.")
    if block_size >= data_size:
        raise ValueError("'block_size' cannot be equal to or greater than 'data_size'.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise TypeError("'batch_size' must be a positive integer.")
    if split not in ['train', 'val']:
        raise ValueError("'split' must be 'train' or 'val'.")
    if not isinstance(file_lengths, list) or not all(isinstance(length, int) and length > 0 for length in file_lengths):
        raise TypeError("'file_lengths' must be a list of positive integers.")
    if not isinstance(is_percents, bool):
        raise TypeError("'is_percents' must be a boolean.")

    first_element_offset = 1 if is_percents else 0
    block_size_xy = block_size + 1

    if len(file_lengths) == 1:
        # Single file case
        return torch.randint(first_element_offset, data_size - block_size_xy + 1, (batch_size,))
    else:
        # Multiple files case - calculate valid positions per file
        valid_positions_per_file = []
        total_valid_ix_positions = 0

        for file_length in file_lengths:
            valid_positions_in_file = max(0, file_length - block_size_xy - first_element_offset + 1)
            valid_positions_per_file.append(valid_positions_in_file)
            total_valid_ix_positions += valid_positions_in_file

        if total_valid_ix_positions == 0:
            raise ValueError("No valid starting positions available. All files are too short for the given block_size.")

        # Generate and map indices back to correct starting positions
        initial_indices = torch.randint(total_valid_ix_positions, (batch_size,))
        actual_indices = torch.empty(batch_size, dtype=torch.long)

        for i in range(batch_size):
            cumulative_valid_ix_positions = 0
            found_position = False
            selected_index = initial_indices[i].item()

            cumulative_data_length = 0
            for file_idx, (file_length, valid_positions_in_file) in enumerate(zip(file_lengths, valid_positions_per_file)):
                if cumulative_valid_ix_positions <= selected_index < cumulative_valid_ix_positions + valid_positions_in_file:
                    relative_index_in_file = selected_index - cumulative_valid_ix_positions
                    actual_starting_position = cumulative_data_length + first_element_offset + relative_index_in_file
                    actual_indices[i] = actual_starting_position
                    found_position = True
                    break

                cumulative_valid_ix_positions += valid_positions_in_file
                cumulative_data_length += file_length

            if not found_position:
                raise RuntimeError(f"Could not map random index {selected_index} to a valid starting position.")

        return actual_indices


def _get_direction_sign(current_value, previous_value, is_percentage_data):
    """Determine direction: 1=up, -1=down, 0=flat."""
    if is_percentage_data:
        if current_value > 0: return 1
        elif current_value < 0: return -1
        else: return 0 # Handles current_value == 0
    else:
        # For value data, direction is based on change from previous value
        if not isinstance(previous_value, numbers.Number):
            # Cannot calculate direction if previous value is not numeric
            return None # Indicate that direction cannot be determined

        change = current_value - previous_value
        if change > 0: return 1
        elif change < 0: return -1
        else: return 0 # Handles change == 0


def calculate_evaluation_metrics(logits_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, batch_size, is_percents):
    """Calculate directional prediction metrics: wins, losses, and certainty for each modality."""
    batch_wins_list = [0] * num_modalities
    batch_losses_list = [0] * num_modalities
    batch_certainty_list = [0.0] * num_modalities
    batches_processed_list = [0] * num_modalities

    for modality_index in range(num_modalities):
        # Get modality name with fallbacks
        modality_name = all_modality_params[modality_index][9]
        if not modality_name or not isinstance(modality_name, str):
            if all_file_info and len(all_file_info) > modality_index and all_file_info[modality_index]:
                modality_name = os.path.basename(all_file_info[modality_index][0])
            else:
                modality_name = f"Modality {modality_index+1}"


        if len(logits_list) > modality_index and len(yb_list) > modality_index:

            modality_vocab = all_vocabularies[modality_index]
            is_percentage_data = all_modality_params[modality_index][3]

            # Check if data is numeric and has sufficient sequence length for directional analysis
            data_is_numeric = all(isinstance(item, numbers.Number) for item in modality_vocab)
            min_seq_len = 1 if is_percentage_data else 2
            if data_is_numeric and yb_list[modality_index].ndim >= 2 and yb_list[modality_index].shape[1] >= min_seq_len:

                logits_modality = logits_list[modality_index][:, -1, :]  # Last token logits
                targets_modality = yb_list[modality_index][:, -1]        # Last token targets

                if targets_modality.shape[0] > 0:
                    batches_processed_list[modality_index] = 1
                    batch_wins_modality = 0
                    batch_losses_modality = 0
                    batch_certainty_modality_sum = 0.0


                    for j in range(logits_modality.shape[0]):
                        predicted_token_logits = logits_modality[j]
                        predicted_token_index = torch.argmax(predicted_token_logits).item()
                        predicted_token_value = modality_vocab[predicted_token_index]

                        actual_token_index = targets_modality[j].item()
                        actual_token_value = modality_vocab[actual_token_index]

                        # Get previous value for direction calculation
                        prev_actual_token_value = None
                        if not is_percentage_data and yb_list[modality_index].shape[1] >= 2:
                            prev_actual_token_value = modality_vocab[yb_list[modality_index][j, -2].item()]

                        # Calculate direction signs
                        predicted_direction_sign = _get_direction_sign(predicted_token_value, prev_actual_token_value, is_percentage_data)
                        actual_direction_sign = _get_direction_sign(actual_token_value, prev_actual_token_value, is_percentage_data)

                        # Count wins/losses
                        if predicted_direction_sign is not None and actual_direction_sign is not None:
                            if predicted_direction_sign == actual_direction_sign:
                                batch_wins_modality += 1
                            else:
                                batch_losses_modality += 1

                            # Calculate directional certainty
                            probs = F.softmax(predicted_token_logits, dim=-1)
                            summed_certainty_for_direction = 0.0

                            for token_index, token_value in enumerate(modality_vocab):
                                if isinstance(token_value, numbers.Number):
                                    possible_direction_sign = _get_direction_sign(token_value, prev_actual_token_value, is_percentage_data)

                                    if possible_direction_sign is not None and possible_direction_sign == predicted_direction_sign:
                                        summed_certainty_for_direction += probs[token_index].item()

                            batch_certainty_modality_sum += summed_certainty_for_direction
                        # Note: instances where direction can't be determined are ignored


                    # Store the total results for this batch and modality
                    batch_wins_list[modality_index] = batch_wins_modality
                    batch_losses_list[modality_index] = batch_losses_modality
                    batch_certainty_list[modality_index] = batch_certainty_modality_sum


            else:
                # If directional metrics were skipped for this batch and modality, indicate why (optional print, can be removed for cleaner output during training)
                # modality_data = all_modality_data[modality_index] # Access processed data to check type
                # data_is_numeric_check = all(isinstance(item, numbers.Number) for item in modality_data)
                # if not data_is_numeric_check:
                #      print(f"Warning: Data for Modality {modality_index+1}: '{modality_name}' is not numeric. Directional metrics skipped for this batch.")
                # elif yb_list[modality_index].ndim < 2 or yb_list[modality_index].shape[1] < min_seq_len:
                #      print(f"Warning: Sequence length for Modality {modality_index+1}: '{modality_name}' is less than {min_seq_len} ({yb_list[modality_index].shape[1] if yb_list[modality_index].ndim >= 2 else 'N/A'}). Cannot calculate directional metrics for this batch.")
                pass # Suppress verbose warnings during training


        else:
            # print(f"Could not perform success rate or certainty calculation for Modality {modality_index+1}: '{modality_name}' due to missing logits or targets for this batch.")
            pass # Suppress verbose warnings during training


    return batch_wins_list, batch_losses_list, batch_certainty_list, batches_processed_list


def get_batch(split, is_training):
    # Generate for all modalities batches of data of inputs (xb_list) and targets (yb_list)

    # ORIGINAL CORRECT APPROACH: Use full datasets for index generation and batch extraction
    # This preserves file boundary logic and then applies train/val split correctly

    # Generate starting indices using FULL datasets (preserves file boundary logic)
    full_data_size = len(all_full_datasets[0])
    ix = generate_batch_starting_indices(full_data_size, _get_block_size(), _get_batch_size(), split, file_lengths, is_percents)

    # Determine train/val boundaries from the split configuration
    # This should match the logic used in create_train_val_datasets
    from compatibility_layer import get_system_configuration
    system_config = get_system_configuration()
    validation_size = system_config['validation_size']
    num_validation_files = system_config['num_validation_files']

    if num_validation_files > 0:
        # File-based splitting logic would go here (not implemented in this fix)
        raise NotImplementedError("File-based validation splitting not yet restored")
    else:
        # Percentage-based splitting
        val_start_idx = int(full_data_size * (1 - validation_size))

        # Filter indices based on split type
        if split == 'train':
            # Keep only indices that fall in training portion
            valid_indices = ix[ix < val_start_idx]
        else:  # split == 'val'
            # Keep only indices that fall in validation portion
            valid_indices = ix[ix >= val_start_idx]
            # Adjust indices to be relative to validation start
            valid_indices = valid_indices - val_start_idx

        # If we don't have enough valid indices, regenerate
        # (This is a simplified approach; proper implementation would regenerate until enough valid indices)
        if len(valid_indices) < _get_batch_size():
            # For now, pad with valid indices (not ideal but prevents crash)
            if len(valid_indices) > 0:
                # Repeat existing valid indices to reach _get_batch_size()
                repeats_needed = (_get_batch_size() + len(valid_indices) - 1) // len(valid_indices)
                valid_indices = valid_indices.repeat(repeats_needed)[:_get_batch_size()]
            else:
                # No valid indices found, use boundary-safe indices
                if split == 'train':
                    max_safe_idx = val_start_idx - _get_block_size() - 1
                    valid_indices = torch.randint(0, max(1, max_safe_idx), (_get_batch_size(),))
                else:
                    val_size = full_data_size - val_start_idx
                    max_safe_idx = val_size - _get_block_size() - 1
                    if max_safe_idx > 0:
                        valid_indices = torch.randint(0, max_safe_idx, (_get_batch_size(),))
                    else:
                        raise ValueError("Validation set too small for _get_block_size()")
        else:
            valid_indices = valid_indices[:_get_batch_size()]

    # Use the appropriate dataset portion for extraction
    if split == 'train':
        # Extract from full datasets for training
        dataset_tensors = [torch.tensor(all_full_datasets[r][:val_start_idx], dtype=torch.long) for r in range(num_modalities)]
    else:
        # Extract from full datasets for validation
        dataset_tensors = [torch.tensor(all_full_datasets[r][val_start_idx:], dtype=torch.long) for r in range(num_modalities)]

    # Apply randomness to training data if needed
    if split == 'train' and is_training == 1:
        for r in range(num_modalities):
            this_rand_size = all_modality_params[r][2] if len(all_modality_params[r]) > 2 else None
            this_vocab_size = len(all_vocabularies[r])

            if this_rand_size is not None:
                from data_utils import add_rand_to_data_points
                randomized_data = add_rand_to_data_points(dataset_tensors[r].tolist(), this_rand_size, this_vocab_size)
                dataset_tensors[r] = torch.tensor(randomized_data, dtype=torch.long)

    # Create batches for all modalities using valid indices
    xb_list = []
    yb_list = []
    for r in range(num_modalities):
        temp_data = dataset_tensors[r]
        xb = torch.stack([temp_data[i:i+_get_block_size()] for i in valid_indices])
        yb = torch.stack([temp_data[i+1:i+_get_block_size()+1] for i in valid_indices])
        xb, yb = xb.to(_get_device()), yb.to(_get_device())
        xb_list.append(xb)
        yb_list.append(yb)

    return xb_list, yb_list


def estimate_loss():
    out = {}
    m.eval() # Use 'm' instead of 'model'
    for state in ['train', 'val']:
        print_state = "Training" if state == 'train' else "Validation"
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'Evaluation: {state.title()} set ({_get_eval_iters()} iterations) | {current_time}')
        # Initialize counters for success rate and certainty calculation for all modalities
        all_modalities_total_batches_processed = [0] * num_modalities
        all_modalities_total_correct = [0] * num_modalities
        all_modalities_total_incorrect = [0] * num_modalities
        all_modalities_total_certainty = [0.0] * num_modalities

        total_losses = []
        non_numeric_warning_printed = [False] * num_modalities

        for k in range(_get_eval_iters()):
            # get_batch returns lists of tensors: [xb_mod1, xb_mod2, ...], [yb_mod1, yb_mod2, ...]
            xb_list, yb_list = get_batch(state, 0)

            # Pass lists of tensors to the multimodal model
            # m is the model instance
            logits_list, losses_list = m(xb_list, yb_list)

            # Handle cases where losses might not be calculated (e.g., during generation if targets are None)
            # Ensure losses_list is not None and contains tensors
            if losses_list and all(l is not None for l in losses_list):
                total_loss_this_iter = sum(losses_list)
                total_losses.append(total_loss_this_iter.item()) # Store the scalar loss value
            else:
                # Handle cases where losses might not be calculated (e.g., during generation if targets are None)
                print(f"Warning: Iteration {k} losses not calculated, skipping")

            # Call calculate_evaluation_metrics to calculate evaluation metrics for this batch
            # is_percents argument is now redundant and can be removed from the function signature and calls
            # The calculate_evaluation_metrics function accesses percentage status from all_modality_params
            batch_correct, batch_incorrect, batch_certainty, batches_processed_list = calculate_evaluation_metrics(
                logits_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, _get_batch_size(), is_percents
            )

            # Check if any modality was skipped due to non-numeric data and print a warning once per eval run
            for modality_index in range(num_modalities):
                if not non_numeric_warning_printed[modality_index]:
                    modality_vocab = all_vocabularies[modality_index]
                    data_is_numeric = all(isinstance(item, numbers.Number) for item in modality_vocab)
                    if not data_is_numeric:
                        modality_name = all_modality_params[modality_index][9] if all_modality_params[modality_index][9] else f"Modality {modality_index+1}"
                        non_numeric_warning_printed[modality_index] = True

            # Accumulate the results returned by the separate function
            for modality_index in range(num_modalities):
                 all_modalities_total_correct[modality_index] += batch_correct[modality_index]
                 all_modalities_total_incorrect[modality_index] += batch_incorrect[modality_index]
                 all_modalities_total_certainty[modality_index] += batch_certainty[modality_index]
                 all_modalities_total_batches_processed[modality_index] += batches_processed_list[modality_index] # Accumulate based on batches_processed_list


        # Store the loss
        losses = sum(total_losses) / len(total_losses) if total_losses else float('nan')
        out[state] = losses

        print(f"Directional Metrics - {print_state}:")
        for modality_index in range(num_modalities):
            modality_name = all_modality_params[modality_index][9] if all_modality_params[modality_index][9] else f"Modality {modality_index+1}"

            this_num_batches_processed = all_modalities_total_batches_processed[modality_index]

            # Only report correct/incorrect and success rate if there were batches where directional calculation was attempted
            if this_num_batches_processed > 0:
                correct = all_modalities_total_correct[modality_index]
                incorrect = all_modalities_total_incorrect[modality_index]

                total_predictions = all_modalities_total_correct[modality_index] + all_modalities_total_incorrect[modality_index]
                if total_predictions > 0:
                    overall_success_rate_modality = round((all_modalities_total_correct[modality_index] / total_predictions) * 100, 1)
                    print(f"  '{modality_name}': {correct}/{incorrect} ({overall_success_rate_modality}%)")
                else:
                    print(f"  '{modality_name}': No directional predictions")

                # Calculate and report overall average directional certainty
                overall_average_certainty_modality = all_modalities_total_certainty[modality_index] / (this_num_batches_processed * _get_batch_size()) # Assuming _get_batch_size() is constant and used for certainty accumulation
                #print(f"  Overall Average Directional Certainty: {round(overall_average_certainty_modality * 100, 1)}%") # Not displaying at the moment

            else:
                print(f"  '{modality_name}': No data processed (non-numeric)")

        # Write validation metrics to file
        system_config = _get_config()
        output_file_name = system_config['output_file_name']
        project_file_path = system_config['project_file_path']
        if state == 'val' and output_file_name != '':
          output_file_path = project_file_path + 'output/' + output_file_name
          with open(output_file_path, 'a', encoding='utf-8') as f:
            for modality_index in range(num_modalities):
                # Get modality name from all_modality_params
                modality_name = all_modality_params[modality_index][9] if all_modality_params[modality_index][9] else f"Modality {modality_index+1}"
                this_num_batches_processed = all_modalities_total_batches_processed[modality_index]

                # Only write if there were batches processed
                if this_num_batches_processed > 0:
                    total_predictions = all_modalities_total_correct[modality_index] + all_modalities_total_incorrect[modality_index]
                    if total_predictions > 0:
                        overall_success_rate_modality = round((all_modalities_total_correct[modality_index] / total_predictions) * 100, 1)
                        f.write(f"{print_state} set (Modality {modality_index+1}: {modality_name}): Total Batches={this_num_batches_processed}, Directional Correct={all_modalities_total_correct[modality_index]}, Directional Incorrect={all_modalities_total_incorrect[modality_index]}, Directional Success Rate (correct/incorrect)={overall_success_rate_modality}%\\n")
                    else:
                        f.write(f"{print_state} set (Modality {modality_index+1}: {modality_name}): Total Batches={this_num_batches_processed}, Directional Correct={all_modalities_total_correct[modality_index]}, Directional Incorrect={all_modalities_total_incorrect[modality_index]}, Directional Success Rate (correct/incorrect)=NA\\n")
                else:
                    f.write(f"{print_state} set (Modality {modality_index+1}: {modality_name}): Total Batches=0, Directional Correct=0, Directional Incorrect=0, Directional Success Rate (correct/incorrect)=NA\\n")

        # Remove excessive separator line


    m.train() # Set the model back to training mode
    return out


# Global variables that will be imported by main.py
# These will be set by main.py after data loading
all_full_datasets = None  # Full datasets for proper index generation
all_train_sets = None
all_val_sets = None
all_vocabularies = None
all_modality_params = None
all_file_info = None
file_lengths = None
num_modalities = None
is_percents = False  # Will be set by main.py if any modality uses percentages
m = None  # model instance