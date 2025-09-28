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

# Import configuration utilities
from config_utils import _get_config, _get_device, _get_block_size, _get_batch_size, _get_eval_iters

# Export only the functions that should be publicly available
__all__ = [
    # Batch generation functions
    'generate_batch_starting_indices', 'get_batch',
    # Training utilities
    'estimate_loss', 'calculate_evaluation_metrics'
]


def generate_batch_starting_indices(data_size, block_size, batch_size, split, file_lengths, is_percents):
    '''
    Generates a batch of random starting indices for extracting sequences from split datasets,
    ensuring that sequences don't cross file boundaries and have sufficient length for both
    input and target sequences.

    The function works with train/val splits by using the 'split' parameter to determine
    which portion of the original files corresponds to the current dataset:
    - For 'train': Uses files from the beginning of file_lengths
    - For 'val': Uses files from the end of file_lengths (working backwards)

    Each generated index ensures that both the input sequence (block_size) and target
    sequence (block_size, offset by 1) fit entirely within a single file boundary.

    Args:
        data_size: The total size of the current split dataset ('train' or 'val').
                   Must be a positive integer.
        block_size: The length of input sequences to be extracted (context window size).
                    Must be a positive integer and less than data_size.
                    Note: Actual space needed is block_size + 1 for both input and target.
        batch_size: The number of starting indices to generate.
                    Must be a positive integer.
        split: Specifies which dataset split this is ('train' or 'val').
               Determines which files from file_lengths to use for boundary calculation.
        file_lengths: Complete list of file lengths from the original full dataset.
                      Used to determine file boundaries within the current split.
                      Must be a list of positive integers.
        is_percents: Whether percentage conversion is used for any modality.
                     If True, skips the first element of each file as a starting candidate
                     (since first element = 0 for percentage data).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing random starting indices
                      that respect file boundaries within the current split dataset.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values (non-positive sizes, invalid split,
                    empty file_lengths, block_size >= data_size, or insufficient valid
                    starting positions available).
    '''

    # --- Input Validation ---
    if not isinstance(data_size, int) or data_size <= 0:
        raise TypeError("'data_size' must be a positive integer.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise TypeError("'block_size' must be a positive integer.")
    if block_size >= data_size:
        raise ValueError("'block_size' cannot be equal to or greater than 'data_size'.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise TypeError("'batch_size' must be a positive integer.")
    if not isinstance(split, str) or (split != 'train' and split != 'val'):
        raise ValueError("'split' must be 'train' or 'val'.")
    if not isinstance(file_lengths, list) or not len(file_lengths) >= 1:
        raise TypeError("'file_lengths' must be a list containing at least 1 element.")
    if not isinstance(is_percents, bool):
        raise TypeError("'is_percents' must be a boolean.")


    block_size_xy = block_size + 1  # Space needed for both input and target sequences


    if is_percents:
        # The 1st element in each file will be skipped when generating starting indices
        first_element_offset = 1
    else:
        first_element_offset = 0


    if len(file_lengths) == 1:
        # Single file case - generate indices with boundary checks
        return torch.randint(first_element_offset, data_size - block_size_xy + 1, (batch_size,))


    if len(file_lengths) > 1:
        # When dealing with a combined dataset of multiple files, we need to ensure sequences don't cross file boundaries.
        dataset_file_lengths = []  # File lengths for current split only
        num_files_loaded = len(file_lengths)
        file_size_accum = 0

        for f in range(num_files_loaded):

            if split == 'train':
                this_file_size = file_lengths[f]

            if split == 'val':
                # Work backwards from end of file_lengths (validation set uses end of dataset)
                this_file_size = file_lengths[num_files_loaded - 1 - f]

            file_size_accum += this_file_size

            if file_size_accum <= data_size:
                dataset_file_lengths.append(this_file_size)

            if file_size_accum > data_size:
                # Add the portion of the last loaded file making this data set
                dataset_file_lengths.append(data_size - (file_size_accum - this_file_size))

            if file_size_accum >= data_size:
                if split == 'val':
                    # Now we reverse the order of lengths because we accumulated file sizes from the end of file_lengths backwards
                    dataset_file_lengths.reverse()
                break


        # Calculate the total number of valid starting positions across all relevant files
        # A valid starting position for a file of length L is from first_element_offset to L - block_size_xy.
        # Total valid positions in a file of length L is L - block_size_xy - first_element_offset + 1.
        total_valid_ix_positions = sum(max(0, length - block_size_xy - first_element_offset + 1) for length in dataset_file_lengths)


        # Handle the case where there are no valid starting positions
        if total_valid_ix_positions <= 0:
            raise ValueError(f"No valid starting positions available for the given block size and file lengths.")


        # Generate initial random indices within the range of total valid positions
        initial_indices = torch.randint(total_valid_ix_positions, (batch_size,))

        # Now, map these initial indices back to the correct starting positions in the
        # combined data, ensuring they fall within the valid range of a specific file.
        actual_indices = torch.empty(batch_size, dtype=torch.long)

        for i in range(batch_size):
            cumulative_valid_ix_positions = 0
            found_position = False

            # Iterate through only the relevant file lengths
            for k, length in enumerate(dataset_file_lengths):
                valid_ix_positions_in_this_file = max(0, length - block_size_xy - first_element_offset + 1)

                if initial_indices[i] < cumulative_valid_ix_positions + valid_ix_positions_in_this_file:
                    # Map to actual position within the file
                    position_within_file = initial_indices[i] - cumulative_valid_ix_positions
                    start_of_this_file = sum(dataset_file_lengths[:k])
                    actual_indices[i] = start_of_this_file + position_within_file + first_element_offset

                    found_position = True
                    break

                cumulative_valid_ix_positions += valid_ix_positions_in_this_file



            if not found_position:
                 raise ValueError(f"Could not map initial index {initial_indices[i]} to a valid ix position.")


        return actual_indices


def _get_direction_sign(current_value, previous_value, is_percentage_data):
    """Determine the directional sign of a data point for prediction analysis.

    For percentage data, direction is based on the sign of the current value.
    For regular value data, direction is based on change from previous value.

    Args:
        current_value: The current data point value (numeric).
        previous_value: The previous data point value (numeric or None).
        is_percentage_data: Whether this is percentage change data.

    Returns:
        int or None: 1 for upward direction, -1 for downward direction,
                     0 for flat/no change, None if direction cannot be determined.
    """
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


def calculate_evaluation_metrics(logits_list, xb_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, batch_size, is_percents):
    """Calculate directional prediction metrics for each modality.

    Analyzes model predictions vs actual values to determine directional accuracy.
    Only processes numeric modalities with sufficient sequence length.

    Args:
        logits_list: List of prediction logits tensors, one per modality.
        xb_list: List of input tensors, one per modality.
        yb_list: List of target tensors, one per modality.
        num_modalities: Number of modalities to process.
        all_vocabularies: List of vocabulary lists, one per modality.
        all_modality_params: List of modality parameter tuples.
        all_file_info: List of file information for each modality.
        batch_size: Size of the current batch.
        is_percents: Whether any modality uses percentage data.

    Returns:
        Tuple[List[int], List[int], List[float], List[int]]: Lists containing
        wins, losses, certainty scores, and batches processed counts per modality.
    """
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

                        # Get previous value for direction calculation (last token from input sequence)
                        prev_actual_token_value = None
                        if not is_percentage_data and xb_list[modality_index].shape[1] >= 1:
                            prev_actual_token_value = modality_vocab[xb_list[modality_index][j, -1].item()]

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
    """Generate batches of input and target data for all modalities.

    Creates batches by extracting sequences from the specified dataset split.
    Applies randomness to training data if configured for any modality.

    Args:
        split: Dataset split to use ('train' or 'val').
        is_training: Whether this is for training (1) or evaluation (0).
                     Randomness is only applied when is_training=1.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of input and target
        tensors, one per modality. Each tensor has shape (batch_size, block_size).
    """

    # Create a temporary list for train sets to potentially apply randomness
    temp_all_train_sets_processed = [t for t in all_train_sets]

    for r in range(num_modalities):
        this_rand_size = all_modality_params[r][2]
        this_vocab_size = len(all_vocabularies[r])

        # Randomness would only be applied to training sets (is_training = 1)
        if this_rand_size is not None and is_training == 1:
            # Apply randomness to the temporary list for this modality
            from data_utils import add_rand_to_data_points
            temp_all_train_sets_processed[r] = add_rand_to_data_points(temp_all_train_sets_processed[r], this_rand_size, this_vocab_size)

    # Convert processed training data lists to tensors
    temp_all_train_sets_tensors = [torch.tensor(t, dtype=torch.long) for t in temp_all_train_sets_processed]

    # all_val_sets are already tensors from create_train_val_datasets
    temp_data_list = temp_all_train_sets_tensors if split == 'train' else all_val_sets

    # Generate starting indices for the first modality (assuming all modalities have the same length and structure)
    # (we might need to adjust this if modalities have different structures/lengths)
    data_size = len(temp_data_list[0])
    ix = generate_batch_starting_indices(data_size, _get_block_size(), _get_batch_size(), split, file_lengths, is_percents)

    # Create batches for all modalities
    xb_list = []
    yb_list = []
    for r in range(num_modalities):
        temp_data = temp_data_list[r]
        xb = torch.stack([temp_data[i:i+_get_block_size()] for i in ix])
        yb = torch.stack([temp_data[i+1:i+_get_block_size()+1] for i in ix])
        xb, yb = xb.to(_get_device()), yb.to(_get_device())
        xb_list.append(xb)
        yb_list.append(yb)

    return xb_list, yb_list


def estimate_loss(current_step=None, max_steps=None):
    """Estimate model loss and calculate directional prediction metrics.

    Evaluates the model on both training and validation sets, computing:
    - Average loss per modality
    - Directional prediction accuracy (correct/total predictions)
    - Directional certainty scores

    Writes validation metrics to output file if configured.

    Returns:
        dict: Dictionary with 'train' and 'val' keys containing average losses.
    """
    out = {}
    m.eval() # Use 'm' instead of 'model'
    for state in ['train', 'val']:
        print_state = "Training" if state == 'train' else "Validation"
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        step_info = f'Step {current_step}/{max_steps} | ' if current_step is not None else ''
        batch_calc = f' * {_get_batch_size()} batches = {_get_eval_iters() * _get_batch_size()} total'
        print(f'Evaluation: {step_info}{state.title()} set ({_get_eval_iters()} iterations{batch_calc}) | {current_time}')
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
                logits_list, xb_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, _get_batch_size(), is_percents
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

        print(f"\n📈 DIRECTIONAL METRICS - {print_state.upper()} (Correct/Total)")
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
                    print(f"  ▪ {modality_name:<30}{correct}/{total_predictions} ({overall_success_rate_modality}%)")
                else:
                    print(f"  ▪ {modality_name}: No directional predictions")

                # Calculate and report overall average directional certainty
                overall_average_certainty_modality = all_modalities_total_certainty[modality_index] / (this_num_batches_processed * _get_batch_size()) # Assuming _get_batch_size() is constant and used for certainty accumulation
                #print(f"  Overall Average Directional Certainty: {round(overall_average_certainty_modality * 100, 1)}%") # Not displaying at the moment

            else:
                print(f"  ▪ {modality_name}: No data processed (non-numeric)")

        # Write training and validation metrics to file
        system_config = _get_config()
        output_file_name = system_config['output_file_name']
        project_file_path = system_config['project_file_path']
        if output_file_name != '':
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
                        f.write(f"   📊 {print_state.upper()} - {modality_name}: Correct={all_modalities_total_correct[modality_index]:,} | Incorrect={all_modalities_total_incorrect[modality_index]:,} | Accuracy={overall_success_rate_modality}%\n")
                    else:
                        f.write(f"   📊 {print_state.upper()} - {modality_name}: Correct={all_modalities_total_correct[modality_index]:,} | Incorrect={all_modalities_total_incorrect[modality_index]:,} | Accuracy=N/A\n")
                else:
                    f.write(f"   📊 {print_state.upper()} - {modality_name}: Correct=0 | Incorrect=0 | Accuracy=N/A\n")

            # Add spacing between TRAINING and VALIDATION sections in file
            if state == 'train':
                f.write("\n")

        # Add space between train and val sections
        if state == 'train':
            print()


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