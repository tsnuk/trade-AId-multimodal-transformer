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
from config import device, block_size, batch_size, eval_iters

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
    '''
    Generates a batch of random starting indices for extracting sequences
    of a fixed length (block_size) from the data.

    This function is designed to ensure that sequences don't cross file boundaries
    when the data comes from multiple concatenated files. It respects the structure
    of the original files to maintain data integrity.

    Args:
        data_size: The total size of the data from which to extract sequences.
                   Must be a positive integer.
                   Also, block_size must be less than `data_size`.
        batch_size: The number of starting indices to generate (the batch size).
                    Must be a positive integer.
        split: Indicates whether the data is for 'train' or 'val'. Must be 'train' or 'val'.
        file_lengths: A list of integers representing the length of each loaded file
                      in the order they were loaded. Must be a list of positive integers.
        is_percents: A boolean indicating whether the data represents percentages.
                     If True, the first element of each file is skipped (since it's always 0
                     for percentage data).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the
                      random starting indices within the dataset.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values.
    '''
    # Input validation
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
    if not isinstance(file_lengths, list) or not all(isinstance(length, int) and length > 0 for length in file_lengths):
        raise TypeError("'file_lengths' must be a list of positive integers.")
    if not isinstance(is_percents, bool):
        raise TypeError("'is_percents' must be a boolean.")

    if len(file_lengths) == 1:
        # Single file case, simpler logic
        first_element_offset = 1 if is_percents else 0
        block_size_xy = block_size + 1

        # Generate random starting indices for the single file, ensuring sequences don't cross the file boundary.
        # Ensure the starting index + block_size_xy does not exceed the file length.
        # then generate random starting indices ensuring sequences fit within the data (sequence + block_size_xy).
        # Adjust the range to start from 'first_element_offset'
        return torch.randint(first_element_offset, data_size - block_size_xy + 1, (batch_size,))


    else:
        # Multiple files case
        first_element_offset = 1 if is_percents else 0
        block_size_xy = block_size + 1

        # Calculate valid starting positions for each file
        # A valid starting position allows for a full sequence (block_size_xy elements) within the file
        valid_positions_per_file = []
        total_valid_ix_positions = 0

        for file_length in file_lengths:
            valid_positions_in_file = max(0, file_length - block_size_xy - first_element_offset + 1)
            valid_positions_per_file.append(valid_positions_in_file)
            total_valid_ix_positions += valid_positions_in_file

        if total_valid_ix_positions == 0:
            raise ValueError("No valid starting positions available. All files are too short for the given block_size.")

        # Generate initial random indices within the range of total valid positions
        initial_indices = torch.randint(total_valid_ix_positions, (batch_size,))

        # Now, map these initial indices back to the correct starting positions in the
        # combined data, ensuring they fall within the valid range of a specific file.
        actual_indices = torch.empty(batch_size, dtype=torch.long)

        for i in range(batch_size):
            cumulative_valid_ix_positions = 0
            found_position = False
            selected_index = initial_indices[i].item()

            # Determine which file this index falls into
            cumulative_data_length = 0
            for file_idx, (file_length, valid_positions_in_file) in enumerate(zip(file_lengths, valid_positions_per_file)):
                if cumulative_valid_ix_positions <= selected_index < cumulative_valid_ix_positions + valid_positions_in_file:
                    # The selected index falls within this file's valid range
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
    """
    Determines the direction sign (1 for up, -1 for down, 0 for flat)
    based on the current and previous numeric values and whether the data is percentages.

    Args:
        current_value: The current numeric value.
        previous_value: The previous numeric value (only used if not percentage data).
        is_percentage_data: Boolean indicating if the data represents percentages.

    Returns:
        An integer: 1 for up, -1 for down, 0 for flat.
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


def calculate_evaluation_metrics(logits_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, batch_size, is_percents):
    """
    Calculates success rate (based on predicting the correct direction of change for numeric data)
    and certainty (confidence in the prediction) for each modality from evaluation logits and targets.

    This function processes the output of the model (logits) and the actual target values (yb)
    for a given batch during evaluation. It iterates through each modality and, for numeric data,
    determines if the model's prediction for the last token in a sequence correctly predicted
    the direction of change compared to the previous token.

    For the success rate calculation, *any* predicted direction (up, down, or flat) is compared
    against the actual direction. Wins are counted when the predicted direction matches the
    actual direction (e.g., both up, both down, or both flat). Losses are counted when they do not match.

    The certainty metric represents the model's confidence in the predicted *direction* of change for
    the last token. It is calculated by summing the probabilities (from the softmax of the last token's
    logits) of all possible next tokens that fall within the same direction (up, down, or flat) as the
    single token with the highest predicted probability.

    For applications like stock price prediction, accurately forecasting the direction of movement can
    be highly beneficial, as predicting the exact next price can be significantly more challenging and
    isn't always necessary.

    Note that unusually high directional certainty rates may occur, especially in early training stages
    or with certain data distributions, even if the overall success rate is near chance. This indicates
    the model is strongly skewed towards predicting a particular direction, regardless of accuracy.

    These directional metrics provide additional insights into the model's behavior during evaluation.
    The model's learning process, however, is driven solely by minimizing the calculated loss.

    The function accumulates these metrics across the batch and reports the results for each modality.

    Args:
        logits_list: A list of tensors, one for each modality, containing the logits
                     for each token in the batch. Shape: List of [(batch_size, block_size, vocab_size)]
        yb_list: A list of tensors, one for each modality, containing the target token
                 indices for each token in the batch. Shape: List of [(batch_size, block_size)]
        num_modalities: The total number of modalities being processed. (int)
        all_vocabularies: A list of lists, where each inner list is the vocabulary
                          (unique elements) for a specific modality, sorted in ascending order.
        all_modality_params: A list of lists, where each inner list contains the processing parameters
                             for a specific modality, in the format [num_whole_digits, decimal_places,
                             rand_size, cross_attend, convert_to_percents, num_bins, modality_name].
        all_file_info: A list of lists, where each inner list contains the file information
                       for a specific modality, in the format [file1_name, data1_length, ...].
        batch_size: The number of sequences processed in parallel in each batch. (int)
        is_percents: A boolean indicating whether the data being evaluated is in percentage form. (bool)

    Returns:
        A tuple containing four lists:
        - batch_wins_list: List of wins for each modality in the current batch.
        - batch_losses_list: List of losses for each modality in the current batch.
        - batch_certainty_list: List of total certainty for each modality in the current batch.
        - batches_processed_list: List of 1 (if batch was processed for directional metrics) or 0 for each modality.
    """
    batch_wins_list = [0] * num_modalities
    batch_losses_list = [0] * num_modalities
    batch_certainty_list = [0.0] * num_modalities
    batches_processed_list = [0] * num_modalities # To indicate if directional metrics were calculated for this modality in this batch


    for modality_index in range(num_modalities):
        # Get modality name from all_modality_params
        modality_name = all_modality_params[modality_index][9]
        # Use the first file name as a fallback if modality_name is not provided or is empty string
        if not modality_name or not isinstance(modality_name, str):
            # Get the name of the first file loaded for this modality from all_file_info
            # all_file_info[modality_index][0] is the name of the first file
            if all_file_info and len(all_file_info) > modality_index and all_file_info[modality_index]:
                modality_name = os.path.basename(all_file_info[modality_index][0])
            else:
                modality_name = f"Modality {modality_index+1}" # Fallback if no file info is available


        if len(logits_list) > modality_index and len(yb_list) > modality_index:

            modality_vocab = all_vocabularies[modality_index]
            is_percentage_data = all_modality_params[modality_index][3] # Get percentage flag from params

            # Check if the modality data is numeric and sequence length is sufficient for directional calculations
            data_is_numeric = all(isinstance(item, numbers.Number) for item in modality_vocab)
            min_seq_len = 1 if is_percentage_data else 2
            if data_is_numeric and yb_list[modality_index].ndim >= 2 and yb_list[modality_index].shape[1] >= min_seq_len:

                logits_modality = logits_list[modality_index][:, -1, :] # Logits for the last token
                targets_modality = yb_list[modality_index][:, -1] # Target index for the last token

                if targets_modality.shape[0] > 0: # Ensure there are elements in the target batch
                    batches_processed_list[modality_index] = 1 # Mark this modality as processed for directional metrics in this batch
                    batch_wins_modality = 0
                    batch_losses_modality = 0
                    batch_certainty_modality_sum = 0.0


                    for j in range(logits_modality.shape[0]): # Iterate through each sequence in the batch
                        predicted_token_logits = logits_modality[j]
                        predicted_token_index = torch.argmax(predicted_token_logits).item()
                        predicted_token_value = modality_vocab[predicted_token_index]

                        actual_token_index = targets_modality[j].item()
                        actual_token_value = modality_vocab[actual_token_index]

                        # Get previous actual value if needed for non-percentage data
                        prev_actual_token_value = None
                        if not is_percentage_data and yb_list[modality_index].shape[1] >= 2:
                            prev_actual_token_value = modality_vocab[yb_list[modality_index][j, -2].item()]


                        # Determine Predicted and Actual Direction Signs using helper function
                        predicted_direction_sign = _get_direction_sign(predicted_token_value, prev_actual_token_value, is_percentage_data)
                        actual_direction_sign = _get_direction_sign(actual_token_value, prev_actual_token_value, is_percentage_data)


                        # --- Count wins/losses based on direction signs ---
                        # Only count if both predicted and actual directions could be determined
                        if predicted_direction_sign is not None and actual_direction_sign is not None:
                            if predicted_direction_sign == actual_direction_sign:
                                batch_wins_modality += 1
                            else:
                                batch_losses_modality += 1


                            # --- Directional Certainty Calculation for this batch instance (j) ---
                            probs = F.softmax(predicted_token_logits, dim=-1)
                            summed_certainty_for_direction = 0.0

                            # Iterate through all possible next tokens in the vocabulary
                            for token_index, token_value in enumerate(modality_vocab):
                                if isinstance(token_value, numbers.Number): # Only consider numeric vocabulary values for certainty
                                    # Determine the direction sign of this possible token relative to the relevant previous value
                                    possible_direction_sign = _get_direction_sign(token_value, prev_actual_token_value, is_percentage_data)

                                    # Check if this possible token's direction sign matches the *predicted* direction sign for this batch instance (j)
                                    if possible_direction_sign is not None and possible_direction_sign == predicted_direction_sign:
                                        summed_certainty_for_direction += probs[token_index].item()

                            # Add the calculated certainty for this batch instance (j) to the batch total
                            batch_certainty_modality_sum += summed_certainty_for_direction
                        else:
                            # If direction could not be determined for this instance, it's not counted for wins/losses or certainty
                            pass


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
    ix = generate_batch_starting_indices(full_data_size, block_size, batch_size, split, file_lengths, is_percents)

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
        if len(valid_indices) < batch_size:
            # For now, pad with valid indices (not ideal but prevents crash)
            if len(valid_indices) > 0:
                # Repeat existing valid indices to reach batch_size
                repeats_needed = (batch_size + len(valid_indices) - 1) // len(valid_indices)
                valid_indices = valid_indices.repeat(repeats_needed)[:batch_size]
            else:
                # No valid indices found, use boundary-safe indices
                if split == 'train':
                    max_safe_idx = val_start_idx - block_size - 1
                    valid_indices = torch.randint(0, max(1, max_safe_idx), (batch_size,))
                else:
                    val_size = full_data_size - val_start_idx
                    max_safe_idx = val_size - block_size - 1
                    if max_safe_idx > 0:
                        valid_indices = torch.randint(0, max_safe_idx, (batch_size,))
                    else:
                        raise ValueError("Validation set too small for block_size")
        else:
            valid_indices = valid_indices[:batch_size]

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
        xb = torch.stack([temp_data[i:i+block_size] for i in valid_indices])
        yb = torch.stack([temp_data[i+1:i+block_size+1] for i in valid_indices])
        xb, yb = xb.to(device), yb.to(device)
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
        print(f'\\nEvaluating {state} set ({eval_iters} iterations)... Current time: {current_time}')
        # Initialize counters for success rate and certainty calculation for all modalities
        all_modalities_total_batches_processed = [0] * num_modalities
        all_modalities_total_correct = [0] * num_modalities
        all_modalities_total_incorrect = [0] * num_modalities
        all_modalities_total_certainty = [0.0] * num_modalities

        total_losses = []
        non_numeric_warning_printed = [False] * num_modalities

        for k in range(eval_iters):
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
                print(f"Warning: Losses not calculated for iteration {k} in state {state}. Skipping loss recording for this iter.")

            # Call calculate_evaluation_metrics to calculate evaluation metrics for this batch
            # is_percents argument is now redundant and can be removed from the function signature and calls
            # The calculate_evaluation_metrics function accesses percentage status from all_modality_params
            batch_correct, batch_incorrect, batch_certainty, batches_processed_list = calculate_evaluation_metrics(
                logits_list, yb_list, num_modalities, all_vocabularies, all_modality_params, all_file_info, batch_size, is_percents
            )

            # Check if any modality was skipped due to non-numeric data and print a warning once per eval run
            for modality_index in range(num_modalities):
                if not non_numeric_warning_printed[modality_index]:
                    modality_vocab = all_vocabularies[modality_index]
                    data_is_numeric = all(isinstance(item, numbers.Number) for item in modality_vocab)
                    if not data_is_numeric:
                        modality_name = all_modality_params[modality_index][9] if all_modality_params[modality_index][9] else f"Modality {modality_index+1}"
                        print(f"Note: Data for Modality {modality_index+1}: '{modality_name}' is not numeric. Directional metrics will be skipped.")
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

        print(f"\\n-------  Directional Metrics Summary  -------")
        print(f"\\n{print_state} set:")
        for modality_index in range(num_modalities):
            modality_name = all_modality_params[modality_index][9] if all_modality_params[modality_index][9] else f"Modality {modality_index+1}"

            print(f"\\nModality {modality_index+1}: '{modality_name}'")
            this_num_batches_processed = all_modalities_total_batches_processed[modality_index]

            # Only report correct/incorrect and success rate if there were batches where directional calculation was attempted
            if this_num_batches_processed > 0:
                print(f'  Total batches processed (iters x batches): {this_num_batches_processed * batch_size}')
                print(f'  Correct direction predictions: {all_modalities_total_correct[modality_index]}')
                print(f'  Incorrect direction predictions: {all_modalities_total_incorrect[modality_index]}')

                total_predictions = all_modalities_total_correct[modality_index] + all_modalities_total_incorrect[modality_index]
                if total_predictions > 0:
                    overall_success_rate_modality = round((all_modalities_total_correct[modality_index] / total_predictions) * 100, 1)
                    print(f'  Overall directional success rate (correct/incorrect): {overall_success_rate_modality}%')
                else:
                    print(f'  Overall directional success rate: NA (No movements predicted or occurred in counted batches)')

                # Calculate and report overall average directional certainty
                overall_average_certainty_modality = all_modalities_total_certainty[modality_index] / (this_num_batches_processed * batch_size) # Assuming batch_size is constant and used for certainty accumulation
                #print(f"  Overall Average Directional Certainty: {round(overall_average_certainty_modality * 100, 1)}%") # Not displaying at the moment

            else:
                print(f'  No batches processed for directional metrics (likely non-numeric data)')

        # Write validation metrics to file
        from config import output_file_name, project_file_path
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

        print('\\n\\n-----------------------------------\\n')


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