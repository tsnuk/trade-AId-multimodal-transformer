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
    'range_numeric_data', 'bin_numeric_data', 'calculate_percent_changes', 'add_rand_to_data_points',
    # Utility functions
    'report_non_numeric_error'
]


def load_file_data(input_info):
    """
    Reads data from a specified file or folder and extracts data from a
    given column. This data will be used to form a single modality for the
    multimodal processing framework. Handles CSV and TXT formats with optional header,
    attempting both comma and semicolon delimiters.

    Optionally, the extracted numeric data can be converted into percentage changes.

    Args:
        input_info: A list containing 10 elements:
            1. Path to a data file or a folder containing data files. Files
               must have '.csv' or '.txt' extensions (str).
            2. The 1-based index of the column to extract data from (int).
            3. Boolean indicating if the data column has a header row (bool).
            4. Boolean indicating if the data should be converted to percentage changes (bool or None).
            5. Number of whole digits (int or None, for ranging - not used in this function).
            6. Number of decimal places (int or None - not used in this function).
            7. Bin data (int or None, for binning - not used in this function).
            8. Randomness size (int or None, for data augmentation - not used in this function).
            9. Cross-attention status (bool or None, for model configuration - not used in this function).
            10. Modality name (str or None - not used in this function).


    Returns:
        A tuple containing:
        - A list of the loaded data points (can be of various data types: numeric, string, ...).
          If 'convert_to_percentages' is True, this list will contain float percentage changes.
        - A list containing the names and lengths of the loaded files:
            [file1_name (str), file1_length (int), file2_name (str), file2_length (int), ...]

    Raises:
        TypeError: If input_info or its elements are not of the expected types.
        ValueError: If input_info is empty or does not contain exactly 10 elements,
                    if the data path is invalid or no supported files are found,
                    or if the specified column does not exist.
        RuntimeError: If an unexpected error occurs during file loading.
        ZeroDivisionError: If attempting to calculate percentage change with a zero value.
    """

    if not isinstance(input_info, list):
        raise TypeError("'input_info' must be a list.")
    # Validation to check for 10 elements
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

    # Validate the convert_to_percentages flag
    convert_to_percentages = input_info[3]
    if not (isinstance(convert_to_percentages, bool) or convert_to_percentages is None):
        raise TypeError(f"Element 4 (convert to percentages) of 'input_info' must be a boolean or None, but got {type(convert_to_percentages).__name__}.")

    # Get modality name (element 10)
    modality_name = input_info[9]
    if not (isinstance(modality_name, str) or modality_name is None):
        raise TypeError(f"Element 10 (modality name) of 'input_info' must be a string or None, but got {type(modality_name).__name__}.")

    data_file_paths = []
    if os.path.isdir(data_path):
        # Path to a folder
        load_from = "folder"
        data_file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and (f.endswith('.csv') or f.endswith('.txt'))]
        if not data_file_paths:
            raise ValueError(f"No CSV or TXT files found in folder '{data_path}'.")

    elif os.path.isfile(data_path):
        # Path to a file
        load_from = "file"
        if not (data_path.endswith('.csv') or data_path.endswith('.txt')):
            raise ValueError(f"The specified file '{data_path}' is not a CSV or TXT file.")
        data_file_paths.append(data_path)


    # Read the datafile/s
    loaded_data = []
    data_info = []

    # This will be used for rounding if data is specified to be converted to percentages (convert_to_percentages is true)
    num_dec_places = input_info[5]

    data_name_from_path = Path(data_path).name
    #prep = "for" if modality_name else "from"
    #print(f"  Loading data {prep}: '{modality_name if modality_name else data_name_from_path}'") # Print modality name if provided
    print(f"  Loading data from {load_from}: '{data_name_from_path}'") # Print modality name if provided


    for full_path in data_file_paths:
        filename = os.path.basename(full_path)
        df = pd.DataFrame() # Initialize empty DataFrame
        read_successful = False

        # Try reading with comma delimiter first
        try:
            df = pd.read_csv(full_path, delimiter=',', engine='python', header=None, skiprows=1 if has_header else 0)
            if not df.empty:
                read_successful = True
                print(f'  Successfully read file: {filename}')
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            last_error = e # Store the last error

        # If not successful, try reading with semicolon delimiter
        if not read_successful:
            try:
                df = pd.read_csv(full_path, delimiter=';', engine='python', header=None, skiprows=1 if has_header else 0)
                if not df.empty:
                    read_successful = True
                    print(f'  Successfully read file: {filename}')
            except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
                last_error = e # Store the last error


        # If after trying both delimiters, the DataFrame is still empty or read was not successful
        if not read_successful or df.empty:
            error_message = f"Failed to load data from file '{filename}' after trying both comma and semicolon delimiters."
            if 'last_error' in locals(): # Check if an error was caught
                error_message += f" Last error: {last_error}"
            print(error_message)
            raise RuntimeError(error_message)


        if num_data_column > df.shape[1]:
            raise ValueError(f"The specified data column ({num_data_column}) does not exist in file '{filename}'. File has {df.shape[1]} columns.")

        column_data = df.iloc[:, num_data_column - 1]

        # Convert column data to a list
        column_data_list = column_data.tolist()

        # Check if convert_to_percentages is True before processing
        if convert_to_percentages is True:
            # Check if data is numeric before calculating percentages
            data_is_numeric = all(isinstance(item, numbers.Number) for item in column_data_list)
            if not data_is_numeric:
                # Find and report the non-numeric element
                print(f"\\nError: Percentage calculation specified for Modality '{modality_name if modality_name else data_name_from_path}' from file '{filename}', but data is not entirely numeric.")
                report_non_numeric_error(column_data_list, data_info + [filename, len(column_data_list)], modality_name if modality_name else data_name_from_path)

            # Proceed with percentage calculation since data is confirmed to be numeric
            try:
                percentages = calculate_percent_changes(column_data_list, num_dec_places if num_dec_places else 2)
                loaded_data.extend(percentages)
            except ZeroDivisionError as e:
                print(f"\\nError: Division by zero encountered when calculating percentage changes for Modality '{modality_name if modality_name else data_name_from_path}' from file '{filename}'.")
                print(f"This usually occurs when the data contains consecutive zero values.")
                raise e

        else:
            # No percentage conversion, add as-is
            loaded_data.extend(column_data_list)

        # Add filename and data length to file info
        data_info.extend([filename, len(column_data_list)])


    return loaded_data, data_info


def report_non_numeric_error(data_list, file_info, this_modality):
    """
    Finds the first non-numeric element in a data list and raises a ValueError,
    reporting its location, including the file name and approximate element index within that file,
    as well as the element's value and type.

    Args:
        data_list: A list of data points to check for non-numeric elements.
        file_info: A list containing the file information in the format
                   [file1_name, data1_length, file2_name, data2_length, ...].
        this_modality: An integer representing the 1-based index of the modality,
                       or a string representing the name of the modality.

    Raises:
        ValueError: If a non-numeric element is found in the data_list.
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
  """
  Converts a list of data points (numeric or other types) into a numerical
  representation by mapping each unique element to an integer index.

  Args:
    data_points: A list of data points. Can be numeric or other types.

  Returns:
    A tuple containing:
    - A list of integers representing the numerical representation of the input data.
    - A list of the unique elements (vocabulary) sorted in ascending order.
  """

  # Create vocabulary of unique elements
  vocabulary = sorted(list(set(data_points)))

  # Map elements to indices
  data_mapping = {element: index for index, element in enumerate(vocabulary)}

  # Transform data to indices
  transformed_data = [data_mapping[element] for element in data_points]

  return transformed_data, vocabulary


def create_train_val_datasets(numeric_rep_data, val_size, num_val_files, file_lengths):
    """
    Splits a combined list of numerical data into training and validation datasets.

    The splitting is done based on either a specified percentage of the total data
    or by allocating a specified number of the *last* data files loaded
    to the validation set.

    When num_val_files > 0, the last num_val_files loaded files are allocated
    to validation, and validation percentage (val_size) is ignored.

    Args:
        numeric_rep_data: A list of numerical data points representing the combined data.
                          Must be a list containing integers.
        val_size: A float between 0 and 1 representing the portion of the data
                  to allocate to the validation set. Used only if num_val_files is 0.
        num_val_files: An integer specifying the number of the last loaded files
                       to allocate to the validation set. If > 0, overrides val_size.
        file_lengths: A list of integers representing the length of each loaded file
                      in the order they were loaded.

    Returns:
        A tuple containing:
        - train_dataset: The list containing the training data. This list will be converted
                         to a tensor at a later stage (in the get_batch function).
        - val_dataset: The tensor containing the validation data.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values (e.g., inconsistent lengths,
                    invalid val_size, invalid num_val_files).
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

        print(f"  Percentage-based splitting: {val_size*100}% for validation.")

    print(f"    Train set size: {train_num_elements}")
    print(f"    Validation set size: {val_num_elements}")


    train_dataset = numeric_rep_data[:train_num_elements]
    val_dataset = torch.tensor(numeric_rep_data[train_num_elements:], dtype=torch.long)


    return train_dataset, val_dataset


def add_rand_to_data_points(numeric_data, rand_size, vocab_size):
    """
    Introduces small random changes to numeric data for data augmentation.

    To mitigate limited trading data volume compared to language training,
    this function synthetically increases the amount of data by adding a small random value
    within a specified range to each data point. This creates slightly varied
    versions of the original data.

    Args:
      numeric_data: A list or tensor of integers (indices representing the original data).
      rand_size: An integer between 1 and 3 specifying the maximum random value range.
                 The actual random values will be in the range [-rand_size, rand_size].
      vocab_size: The size of the vocabulary, used to ensure augmented values stay within bounds.

    Returns:
      A list or tensor of integers with small random changes applied.

    Raises:
      TypeError: If inputs are not of the expected types.
      ValueError: If inputs are not of the expected values.
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
  """
  Converts numeric data to a specified range by scaling them by factors of 10
  and/or rounds to a specified number of decimal places.

  The purpose is to standardize data magnitude and precision, thereby controlling
  vocabulary size.

  Args:
    numeric_data: A list of numeric data points. Must be a list containing numeric types.
    num_whole_digits: The desired number of whole digits for the ranged data.
                      Must be an integer greater than 0, or None.
    decimal_places: The desired number of decimal places for the ranged data.
                    Must be an integer greater than or equal to 0, or None.

  Returns:
    A list of float numbers that have been ranged and rounded.

  Raises:
    TypeError: If inputs are not of the expected types.
    ValueError: If inputs have invalid values (e.g., empty list,
                negative decimal_places if not None, non-numeric data).
    IndexError: If an element in 'numeric_data' is not a number.
  """

  # Input validation
  if not isinstance(numeric_data, list):
      raise TypeError("'numeric_data' must be a list.")
  if not numeric_data:
      raise ValueError("'numeric_data' cannot be empty.")

  # check for numeric data
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

  # check if any processing is specified
  if num_whole_digits is None and decimal_places is None:
      # Nothing to do, return original data as a list of floats (for consistency)
      return [float(item) for item in numeric_data]

  # If only one of the parameters is specified, treat it as num_whole_digits
  if num_whole_digits is None and decimal_places is not None:
      # User provided only one parameter, treat it as num_whole_digits
      num_whole_digits = decimal_places
      decimal_places = 0  # Apply no rounding

  # From this point, we know that at least num_whole_digits is not None

  for data_point in numeric_data:
    processed_point = data_point

    # Apply ranging if num_whole_digits is specified
    if num_whole_digits is not None:
        # Calculate the magnitude and scaling
        magnitude = len(str(int(abs(processed_point))))
        target_magnitude = num_whole_digits

        if magnitude != target_magnitude:
            scaling_factor = 10 ** (target_magnitude - magnitude)
            processed_point *= scaling_factor

    # Apply rounding if decimal_places is specified
    if decimal_places is not None:
        processed_point = round(processed_point, decimal_places)

    processed_data.append(processed_point)


  return processed_data


def bin_numeric_data(data, num_groups, outlier_percentile=5, exponent=2.0):
    """
    Divides a list of numeric data into a specified number of groups with
    non-uniform ranges, based on an exponential-like distribution, after
    removing outliers using percentiles, handling both positive and negative values symmetrically.

    Args:
        data: A list of numeric data points to be binned. Must be a list
              containing numeric types.
        num_groups: The number of groups to create for positive values (and the same
                    for negative values). Total groups = 2 * num_groups + 1 (including zero).
                    Must be a positive integer.
        outlier_percentile: The percentile threshold for outlier removal
                            (default: 5, removing values below 5th and above 95th percentile).
        exponent: Controls the distribution of group boundaries.
                  A value = 1 creates uniform ranges. Must be a number >= 1. (default: 2.0).

    Returns:
        A list of group assignments (integers) for each data point in the input,
        where each integer represents the group number.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values (e.g., empty data list,
                    non-positive num_groups, invalid outlier_percentile, invalid exponent).
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
        if value >= 0:
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

    print(f"    Binning: {2 * num_groups + 1} groups | Range: [{min(filtered_data):.3f}, {max(filtered_data):.3f}]")

    # Group distribution details removed for cleaner output


    total_assigned = sum(group_assignments.count(i) for i in range(-num_groups, num_groups + 1))
    if total_assigned != len(data):
        print(f"Warning: Total assigned data points ({total_assigned}) does not match input data length ({len(data)}) for {display_modality_name}.")

    return group_assignments


def calculate_percent_changes(data, decimal_places=2):
    """
    Calculates the percentage change between adjacent numeric data points
    and returns a list of the same length by prepending a 0.

    Args:
        data: A list of numeric data points. Must be a list containing numeric types.
        decimal_places: The number of decimal places to round the percentage changes to.
                        If None, the default value of 2 will be used. (default: 2).

    Returns:
        A list of float percentage changes, with the first element being 0.0.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values (e.g., empty data list,
                    negative decimal_places if not None).
        ZeroDivisionError: If an attempt is made to divide by zero when calculating
                           percentage change.
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

    # Calculate percentage changes
    percent_changes = [0.0] # Prepend 0 as the first element (in order to keep the processed data at the same length as the input data)
                            # (this element will later be skipped over when generating batch starting indices)

    for i in range(len(data) - 1):
        current_value = data[i]
        next_value = data[i+1]

        if current_value == 0:
            raise ZeroDivisionError(f"Cannot calculate percentage change: division by zero at index {i} (current value is 0).")

        percentage_change = ((next_value - current_value) / current_value) * 100
        rounded_percentage = round(percentage_change, decimal_places)
        percent_changes.append(rounded_percentage)

    if len(percent_changes) != len(data):
        print(f"Warning: Returned list length ({len(percent_changes)}) does not match input list length ({len(data)}).")

    return percent_changes


def write_initial_run_details(file_path, hyperparams, data_info, modality_configs, run_stats):
    """
    Writes the initial run details (hyperparameters, data info, modality configs)
    to the specified output file.

    Args:
        file_path (str): The full path to the output file.
        hyperparams (dict): A dictionary containing the model hyperparameters.
        data_info (dict): A dictionary containing general data information (e.g., split sizes).
        modality_configs (list): A list of dictionaries, where each dictionary
                                 contains the configuration details for a modality.
        run_stats (dict): A dictionary containing overall run statistics (e.g., number of parameters).
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