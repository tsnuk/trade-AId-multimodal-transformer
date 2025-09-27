"""data_utils.py

Data processing utilities and loading functions for the multimodal transformer system.

Contains all functions related to:
- File loading and data extraction
- Data preprocessing (ranging, binning, percentage calculation)
- Numerical representation and vocabulary creation
- Data augmentation and randomness

Extracted from mm_final_4.py for better code organization.
"""

import pandas as pd
import numpy as np
import numbers
import math
import os
from pathlib import Path

# Export only the functions that should be publicly available
__all__ = [
    # Main data loading functions
    'load_file_data',
    # Data processing functions
    'numerical_representation', 'create_train_val_datasets',
    # Built-in processing functions
    'range_numeric_data', 'bin_numeric_data', 'convert_to_percent_changes', 'add_rand_to_data_points',
    # Utility functions
    'report_non_numeric_error'
]


def load_file_data(input_info):
    """Load data from CSV/TXT files and extract specified column data.

    When percentage conversion is enabled, values are converted to percentage
    changes between consecutive data points.

    Args:
        input_info: List of 10 elements containing path, column number, header flag,
                   percentage conversion flag, and other processing parameters.

    Returns:
        Tuple of (loaded_data_list, file_info_list) where loaded_data_list contains
        the extracted data points and file_info_list contains filename and length pairs.

    Raises:
        TypeError: If input_info is not a list or elements have wrong types.
        ValueError: If input_info doesn't have 10 elements or path/column are invalid.
        FileNotFoundError: If specified path doesn't exist.
        RuntimeError: If file reading fails after trying both delimiters.
    """

    if not isinstance(input_info, list):
        raise TypeError("'input_info' must be a list.")
    if len(input_info) != 10:
        raise ValueError("'input_info' must contain 10 elements: Path, data column number, header status, convert to percentages status, num whole digits, num dec places, bin data, rand size, cross-attention status, modality name.")

    data_path = input_info[0]
    if not isinstance(data_path, str):
        raise TypeError(f"Element 1 (Path) of 'input_info' must be a string, but got {type(data_path).__name__}.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path '{data_path}' was not found.")

    num_data_column = input_info[1]
    if not isinstance(num_data_column, int):
        raise TypeError(f"Element 2 (data column number) of 'input_info' must be an integer, but got {type(num_data_column).__name__}.")
    if num_data_column < 1:
        raise ValueError("The specified data column number must be greater than or equal to 1.")

    has_header = input_info[2]
    if not isinstance(has_header, bool):
        raise TypeError(f"Element 3 (header status) of 'input_info' must be a boolean, but got {type(has_header).__name__}.")

    convert_to_percentages = input_info[3]
    if not (isinstance(convert_to_percentages, bool) or convert_to_percentages is None):
        raise TypeError(f"Element 4 (convert to percentages) of 'input_info' must be a boolean or None, but got {type(convert_to_percentages).__name__}.")

    modality_name = input_info[9]
    if not (isinstance(modality_name, str) or modality_name is None):
        raise TypeError(f"Element 10 (modality name) of 'input_info' must be a string or None, but got {type(modality_name).__name__}.")

    data_file_paths = []
    if os.path.isdir(data_path):
        load_from = "folder"
        data_file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and (f.endswith('.csv') or f.endswith('.txt'))]
        if not data_file_paths:
            raise ValueError(f"No CSV or TXT files found in folder '{data_path}'.")
    elif os.path.isfile(data_path):
        load_from = "file"
        if not (data_path.endswith('.csv') or data_path.endswith('.txt')):
            raise ValueError(f"The specified file '{data_path}' is not a CSV or TXT file.")
        data_file_paths.append(data_path)

    loaded_data = []
    data_info = []
    num_dec_places = input_info[5]
    data_name_from_path = Path(data_path).name
    print(f"  Loading data from {load_from}: '{data_name_from_path}'")


    for full_path in data_file_paths:
        filename = os.path.basename(full_path)
        df = pd.DataFrame()
        read_successful = False

        # Try comma delimiter first
        try:
            df = pd.read_csv(full_path, delimiter=',', engine='python', header=None, skiprows=1 if has_header else 0)
            if not df.empty:
                read_successful = True
                print(f'  Successfully read file: {filename}')
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            last_error = e

        # Try semicolon delimiter if comma failed
        if not read_successful:
            try:
                df = pd.read_csv(full_path, delimiter=';', engine='python', header=None, skiprows=1 if has_header else 0)
                if not df.empty:
                    read_successful = True
                    print(f'  Successfully read file: {filename}')
            except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
                last_error = e

        if not read_successful or df.empty:
            error_message = f"Failed to load data from file '{filename}' after trying both comma and semicolon delimiters."
            if 'last_error' in locals():
                error_message += f" Last error: {last_error}"
            print(error_message)
            raise RuntimeError(error_message)


        if num_data_column > df.shape[1]:
            raise ValueError(f"The specified data column ({num_data_column}) does not exist in file '{filename}'. File has {df.shape[1]} columns.")

        column_data = df.iloc[:, num_data_column - 1]
        column_data_list = column_data.tolist()

        if convert_to_percentages is True:
            data_is_numeric = all(isinstance(item, numbers.Number) for item in column_data_list)
            if not data_is_numeric:
                print(f"\\nError: Percentage conversion specified for Modality '{modality_name if modality_name else data_name_from_path}' from file '{filename}', but data is not entirely numeric.")
                report_non_numeric_error(column_data_list, data_info + [filename, len(column_data_list)], modality_name if modality_name else data_name_from_path)

            try:
                percentages = convert_to_percent_changes(column_data_list, num_dec_places if num_dec_places else 2)
                loaded_data.extend(percentages)
            except ZeroDivisionError as e:
                print(f"\\nError: Division by zero encountered when calculating percentage changes for Modality '{modality_name if modality_name else data_name_from_path}' from file '{filename}'.")
                print(f"This usually occurs when the data contains consecutive zero values.")
                raise e
        else:
            loaded_data.extend(column_data_list)

        data_info.extend([filename, len(column_data_list)])


    return loaded_data, data_info


def report_non_numeric_error(data_list, file_info, this_modality):
    """Find and report the first non-numeric element with location details.

    Args:
        data_list: List of data points to check for non-numeric elements.
        file_info: List of file information in format [file1_name, file1_length, ...].
        this_modality: Modality name or index for error reporting.

    Raises:
        ValueError: When a non-numeric element is found in the data_list.
    """
    first_non_numeric_index = -1
    non_numeric_value = None
    non_numeric_type = None

    for idx, item in enumerate(data_list):
        if not isinstance(item, numbers.Number):
            first_non_numeric_index = idx
            non_numeric_value = item
            non_numeric_type = type(item).__name__
            break

    if first_non_numeric_index != -1:
        # Determine which file the non-numeric element belongs to
        cumulative_length = 0
        file_name = "Unknown"
        element_index_in_file = first_non_numeric_index

        # file_info is [file1_name, data1_length, file2_name, data2_length, ...]
        for f_idx in range(0, len(file_info), 2):
            current_file_name = file_info[f_idx]
            current_file_length = file_info[f_idx+1]

            if first_non_numeric_index < cumulative_length + current_file_length:
                file_name = current_file_name
                element_index_in_file = first_non_numeric_index - cumulative_length
                break

            cumulative_length += current_file_length

        raise ValueError(
            f"Non-numeric element found in Modality '{this_modality}' at index {first_non_numeric_index} "
            f"(approximately element {element_index_in_file} in file '{file_name}'). "
            f"Element value: '{non_numeric_value}', Element type: {non_numeric_type}. "
            "Data must be entirely numeric for ranging or decimal places processing."
        )
    # Note: If no non-numeric is found, the function will simply return without raising an error.


def numerical_representation(data_points):
    """Convert data points to numerical indices with vocabulary mapping.

    Args:
        data_points: List of data points (can be numeric, strings, or other types).

    Returns:
        Tuple of (transformed_data, vocabulary) where transformed_data is a list of
        integer indices and vocabulary is the sorted list of unique elements.
    """
    vocabulary = sorted(list(set(data_points)))
    data_mapping = {element: index for index, element in enumerate(vocabulary)}
    transformed_data = [data_mapping[element] for element in data_points]
    return transformed_data, vocabulary


def create_train_val_datasets(numeric_rep_data, val_size, num_val_files, file_lengths):
    """Split numerical data into training and validation datasets.

    Args:
        numeric_rep_data: List of numerical data points to split.
        val_size: Float between 0 and 1 representing validation set proportion.
        num_val_files: Integer specifying number of last files for validation.
        file_lengths: List of integers representing length of each loaded file.

    Returns:
        Tuple of (train_dataset, val_dataset) where both are lists of data points.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If inputs have invalid values or inconsistent lengths.
    """

    if not isinstance(numeric_rep_data, (list)):
        raise TypeError("'numeric_rep_data' must be a list.")

    if not isinstance(num_val_files, int) or num_val_files < 0:
        raise TypeError("'num_val_files' must be a non-negative integer.")

    if not isinstance(file_lengths, list) or not all(isinstance(length, int) and length > 0 for length in file_lengths):
        raise TypeError("'file_lengths' must be a list of positive integers.")

    if sum(file_lengths) != len(numeric_rep_data):
        raise ValueError(f"Sum of file_lengths ({sum(file_lengths)}) does not match length of numeric_rep_data ({len(numeric_rep_data)}).")

    if num_val_files > 0:
        # Use file-based splitting
        if num_val_files > len(file_lengths):
            raise ValueError(f"'num_val_files' ({num_val_files}) cannot exceed the number of loaded files ({len(file_lengths)}).")

        # Calculate validation set size based on the last num_val_files
        val_num_elements = sum(file_lengths[-num_val_files:])
        train_num_elements = len(numeric_rep_data) - val_num_elements

        print(f"  File-based splitting: Last {num_val_files} file(s) for validation.")
        print(f"    Files for validation:")

        # Sum the lengths of the last 'num_val_files' file_lengths for the validation set
        start_index = len(file_lengths) - 1
        for j in range(num_val_files):
            val_num_elements += file_lengths[start_index - j]

    else:
        # Use percentage-based splitting
        if not isinstance(val_size, (int, float)) or not (0 < val_size < 1):
            raise ValueError("'val_size' must be a float between 0 and 1 (exclusive).")

        train_num_elements = int(len(numeric_rep_data) * (1 - val_size))
        val_num_elements = len(numeric_rep_data) - train_num_elements

        # Percentage-based splitting info moved to main.py for better organization
        pass

    # Train/validation set sizes info moved to main.py for better organization


    train_dataset = numeric_rep_data[:train_num_elements]
    val_dataset = torch.tensor(numeric_rep_data[train_num_elements:], dtype=torch.long)


    return train_dataset, val_dataset


def add_rand_to_data_points(numeric_data, rand_size, vocab_size):
    """Introduce small random changes to numeric data for data augmentation.

    Args:
        numeric_data: List or tensor of integers (indices representing original data).
        rand_size: Integer between 1 and 3 specifying maximum random value range.
        vocab_size: Size of vocabulary to ensure augmented values stay within bounds.

    Returns:
        List or tensor of integers with small random changes applied.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If inputs have invalid values or numeric_data is empty.
    """

    # if numeric_data was input as a tensor, then temporarily turn it into a list
    if isinstance(numeric_data, torch.Tensor):
        numeric_data = numeric_data.tolist()
        numeric_data_is_a_tensor = True
    else:
        numeric_data_is_a_tensor = False

    # Input validation for numeric_data
    if not isinstance(numeric_data, list):
        raise TypeError("numeric_data must be a list or a tensor.")
    if not numeric_data:
        raise ValueError("numeric_data cannot be empty.")
    for i, item in enumerate(numeric_data):
        if not isinstance(item, numbers.Number):
            raise ValueError(f"All elements in numeric_data must be numeric. Element at index {i} is {type(item).__name__}: '{item}'.")

    # Input validation for rand_size
    if not isinstance(rand_size, (int, type(None))):
        raise TypeError("rand_size must be an integer or None.")
    if rand_size is not None and (rand_size < 1 or rand_size > 3):
        raise ValueError("rand_size must be an integer between 1 and 3, or None.")

    # Input validation for vocab_size
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise TypeError("vocab_size must be a positive integer.")

    if rand_size is None:
        # Return the original data unchanged
        if numeric_data_is_a_tensor:
            return torch.tensor(numeric_data, dtype=torch.long)
        else:
            return numeric_data

    rand_list = [0]

    for r in range(rand_size):
        rand_list.extend([r+1, -(r+1)])

    for n in range(len(numeric_data)):
        # Check if adding the maximum possible random value still keeps the element within vocabulary bounds
        if max(rand_list) < numeric_data[n] < vocab_size - max(rand_list):
            rand_value = random.choice(rand_list)
            numeric_data[n] += rand_value

    # turn numeric_data back to a tensor
    if numeric_data_is_a_tensor:
        numeric_data = torch.tensor(numeric_data, dtype=torch.long)


    return numeric_data


def range_numeric_data(numeric_data, num_whole_digits, decimal_places):
    """Scale numeric data by factors of 10 and round to specified precision.

    Args:
        numeric_data: List of numeric data points to process.
        num_whole_digits: Target number of whole digits, or None.
        decimal_places: Target number of decimal places, or None.

    Returns:
        List of processed numeric values as floats.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If numeric_data is empty or contains non-numeric values.
    """
    # Input validation
    if not isinstance(numeric_data, list):
        raise TypeError("'numeric_data' must be a list.")
    if not numeric_data:
        raise ValueError("'numeric_data' cannot be empty.")

    for i, item in enumerate(numeric_data):
        if not isinstance(item, numbers.Number):
            raise ValueError(f"All elements in 'numeric_data' must be numeric. Element at index {i} is {type(item).__name__}: '{item}'.")

    if num_whole_digits is not None:
        if not isinstance(num_whole_digits, int) or num_whole_digits <= 0:
            raise TypeError("'num_whole_digits' must be a positive integer or None.")

    if decimal_places is not None:
        if not isinstance(decimal_places, int) or decimal_places < 0:
            raise TypeError("'decimal_places' must be a non-negative integer or None.")

    processed_data = []

    # Check if any processing is specified
    if num_whole_digits is None and decimal_places is None:
        return [float(item) for item in numeric_data]

    # If only decimal_places specified, treat as num_whole_digits
    if num_whole_digits is None and decimal_places is not None:
        num_whole_digits = decimal_places
        decimal_places = 0

    for data_point in numeric_data:
        processed_point = data_point

        # Apply ranging if num_whole_digits is specified
        if num_whole_digits is not None:
            magnitude = len(str(int(abs(processed_point))))
            target_magnitude = num_whole_digits

            if magnitude != target_magnitude:
                scaling_factor = 10 ** (target_magnitude - magnitude)
                processed_point *= scaling_factor

        # Apply rounding if decimal_places is specified
        if decimal_places is not None:
            processed_point = round(processed_point, decimal_places)

            # Handle edge case: if rounding pushed value outside intended range, cap it
            if num_whole_digits is not None:
                max_allowed = (10 ** num_whole_digits) - (10 ** -decimal_places)
                if processed_point >= 10 ** num_whole_digits:
                    processed_point = max_allowed

        processed_data.append(processed_point)

    return processed_data


def bin_numeric_data(data, num_groups, outlier_percentile=5, exponent=2.0):
    """Divide numeric data into groups with exponential distribution after outlier removal.

    Args:
        data: List of numeric data points to bin.
        num_groups: Number of groups for positive values (total groups = 2*num_groups + 1).
        outlier_percentile: Percentile threshold for outlier removal (default: 5).
        exponent: Controls distribution of group boundaries (default: 2.0).

    Returns:
        List of group assignments (integers) for each data point.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If data is empty or parameters are invalid.
    """
    # Input validation
    if not isinstance(data, list) or not data:
        raise ValueError("'data' must be a non-empty list.")
    for i, item in enumerate(data):
        if not isinstance(item, numbers.Number):
            raise ValueError(f"All elements in 'data' must be numeric. Element at index {i} is {type(item).__name__}: '{item}'.")

    if not isinstance(num_groups, int) or num_groups <= 0:
        raise ValueError("'num_groups' must be a positive integer.")
    if not isinstance(outlier_percentile, (int, float)) or not (0 <= outlier_percentile <= 50):
        raise ValueError("'outlier_percentile' must be a number between 0 and 50.")
    if not isinstance(exponent, (int, float)) or exponent < 1:
        raise ValueError("'exponent' must be a number >= 1.")

    # Remove outliers based on percentiles
    lower_percentile = np.percentile(data, outlier_percentile)
    upper_percentile = np.percentile(data, 100 - outlier_percentile)
    filtered_data = [x for x in data if lower_percentile <= x <= upper_percentile]

    if not filtered_data:
        raise ValueError("All data points were filtered out as outliers.")

    # Determine the maximum absolute value for symmetric range creation
    max_abs_value = max(abs(min(filtered_data)), abs(max(filtered_data)))

    # Generate positive group boundaries
    positive_group_boundaries = [0.0] # Start from zero
    for i in range(1, num_groups + 1):
        normalized_i = i / num_groups
        scaled_position = normalized_i**exponent
        boundary_value = scaled_position * max_abs_value
        positive_group_boundaries.append(boundary_value)

    # Generate negative group boundaries (mirror of positive boundaries)
    negative_group_boundaries = [-boundary for boundary in reversed(positive_group_boundaries[1:])] + [0.0]

    # Display modality name for output clarity
    display_modality_name = "this modality"

    # Assign each data point to a group
    group_assignments = []
    for value in data:
        if value > 0:
            # Find the group index for positive values
            group_index = 0
            for j in range(num_groups):
                 # Check if value is within the boundary range [boundary_low, boundary_high)
                 if value >= positive_group_boundaries[j] and value < positive_group_boundaries[j+1]:
                     group_index = j + 1  # Group numbers start from 1
                     break
            else:
                 # Handle the edge case for the maximum value (assign to the last group)
                 group_index = num_groups

            group_assignments.append(group_index)
        elif value == 0:
            # Assign zero values to bin 0
            group_assignments.append(0)
        else:
            # Find the group index for negative values
            # Iterate through negative boundaries (from most negative towards zero)
            # Note: negative_group_boundaries is in increasing order (e.g., [-100, -50, -20, 0])
            for j in range(num_groups):
                 # Check if value is within the boundary range [boundary_low, boundary_high)
                 if value >= negative_group_boundaries[j] and value < negative_group_boundaries[j+1]:
                     group_index = -(num_groups - j)  # Negative group numbers
                     break
            else:
                 # Handle the edge case for the most negative value (assign to the most negative group)
                 group_index = -num_groups

            group_assignments.append(group_index)

    # Display group information (optional, for debugging/transparency)
    group_counts = {}
    for assignment in group_assignments:
        group_counts[assignment] = group_counts.get(assignment, 0) + 1


    # Display binning statistics
    print(f"      → Binning statistics:")

    # Display negative bins (from most negative to least negative)
    for i in range(-num_groups, 0):
        if i in group_counts:
            j = num_groups + i  # Convert to boundary index
            lower_bound = negative_group_boundaries[j]
            upper_bound = negative_group_boundaries[j + 1] if j + 1 < len(negative_group_boundaries) else 0
            count = group_counts[i]

            if i == -num_groups:  # Most negative bin contains outliers
                print(f"        Bin {i}: (-∞, {upper_bound:.3f}) - {count} elements")
            else:
                print(f"        Bin {i}: [{lower_bound:.3f}, {upper_bound:.3f}) - {count} elements")

    # Display zero bin
    if 0 in group_counts:
        count = group_counts[0]
        print(f"        Bin  0: [0.000, 0.000] - {count} elements")

    # Display positive bins
    for i in range(1, num_groups + 1):
        if i in group_counts:
            lower_bound = positive_group_boundaries[i - 1]
            upper_bound = positive_group_boundaries[i] if i < len(positive_group_boundaries) else float('inf')
            count = group_counts[i]

            if i == num_groups:  # Most positive bin contains outliers
                print(f"        Bin {i:2d}: [{lower_bound:.3f}, +∞) - {count} elements")
            else:
                print(f"        Bin {i:2d}: [{lower_bound:.3f}, {upper_bound:.3f}) - {count} elements")


    # Verify all data points were assigned to bins
    total_assigned = sum(group_counts.values())
    if total_assigned != len(data):
        print(f"        ⚠️  Warning: Total assigned elements ({total_assigned}) ≠ input data length ({len(data)})")
    else:
        print(f"        ✓ All {len(data)} elements successfully assigned to bins")

    return group_assignments


def convert_to_percent_changes(data, decimal_places=2):
    """Convert data to percentage changes between adjacent data points (backward-looking).

    Calculates percentage change from previous value to current value:
    ((current - previous) / previous) * 100

    Args:
        data: List of numeric data points.
        decimal_places: Number of decimal places to round to (default: 2).

    Returns:
        List of percentage changes with first element as 0.0.
        Length matches input data length for consistent batch generation.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If data is empty or contains non-numeric values.
        ZeroDivisionError: If division by zero occurs during calculation.
    """
    # Input validation
    if not isinstance(data, list) or not data:
        raise ValueError("'data' must be a non-empty list.")
    for i, item in enumerate(data):
        if not isinstance(item, numbers.Number):
            raise ValueError(f"All elements in 'data' must be numeric. Element at index {i} is {type(item).__name__}: '{item}'.")

    if decimal_places is not None:
        if not isinstance(decimal_places, int) or decimal_places < 0:
            raise ValueError("'decimal_places' must be a non-negative integer or None.")
    else:
        decimal_places = 2 # Default value

    # Convert to percentage changes (backward-looking)
    percent_changes = [0.0] # First element is always 0 (no previous value to compare against)
                            # (this element will later be skipped over when generating batch starting indices)

    for i in range(1, len(data)):
        current_value = data[i]
        previous_value = data[i-1]

        if previous_value == 0:
            raise ZeroDivisionError(f"Cannot calculate percentage change: previous value is zero at index {i-1}.")

        percentage_change = ((current_value - previous_value) / previous_value) * 100
        rounded_percentage = round(percentage_change, decimal_places)
        percent_changes.append(rounded_percentage)

    if len(percent_changes) != len(data):
        print(f"Warning: Returned list length ({len(percent_changes)}) does not match input list length ({len(data)}).")

    return percent_changes


def write_initial_run_details(file_path, hyperparams, data_info, modality_configs, run_stats):
    """Write initial run details to specified output file.

    Args:
        file_path: Full path to the output file.
        hyperparams: Dictionary containing model hyperparameters.
        data_info: Dictionary containing general data information (e.g., split sizes).
        modality_configs: List of dictionaries with modality configuration details.
        run_stats: Dictionary containing overall run statistics (e.g., number of parameters).
    """
    if file_path: # Only write if a file path is provided
        with open(file_path, 'a', encoding='utf-8') as f:
            from datetime import datetime
            now = datetime.now()
            current_time_date = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\\n\\n{current_time_date}\\n")
            f.write("\\nModel Settings and Data Information:\\n")

            # Write hyperparameters
            f.write("Hyperparameters:\\n")
            for key, value in hyperparams.items():
                f.write(f"  {key}: {value}\\n")

            # Write run statistics
            f.write("\\nRun Statistics:\\n")
            for key, value in run_stats.items():
                f.write(f"  {key}: {value}\\n")

            # Write data information
            f.write("\\nData Information:\\n")
            for key, value in data_info.items():
                f.write(f"  {key}: {value}\\n")

            # Write modality configurations
            f.write("\\nInput Schemas (Modality Configurations):\\n")
            for i, config in enumerate(modality_configs):
                f.write(f"  Modality {i+1}:\\n")
                for key, value in config.items():
                    f.write(f"    {key}: {value}\\n")
            f.write("\\n")


# Add required imports that may be missing
import torch
import random