"""config.py

Legacy configuration support for programmatic input schemas.

This module provides backward compatibility for programmatic configuration
when YAML configuration files are not present. Contains input schema definitions
and conditionally loads hyperparameters when config.yaml is missing.
"""

import torch
from datetime import datetime
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

"""Programmatic Input Schema Definitions"""

'''

                                     -------  I N P U T   D A T A  -------


  'input_schema' is a python list in which the user specifies the location from which to retrieve data- path and column number within the data file,
  along with additional info regarding the data (described below), in order for it to be trained on by the system.
  Each input_schema contains/details information designed to construct one modality, and the system is designed to handle
  and process multiple modalities applying selective cross-attention in order to find connections and patterns between modalities.


  Notes:
    1. 10 empty input_schemas are provided below, available to be filled out by you, and used by the system.
       You can leave any unused input_schema as is - an empty list.
       You can also add additional input_schemas beyond the ones provided, according to your needs.
       (you may need to find a balance between too many and too few input_schemas, as overdoing it may compromise results).
    2. All input_schemas should be synchronized to the same dates and times (and ideally be of the same length),
       that is, first elements of all input_schemas would be of the same date and time, all second elements would be of the same date and time, etc.
    3. Please note that the variable 'num_input_schemas' must match the number of existing input_schema lists (both used and empty).
       If you add additional input_schema lists please make sure to modify num_input_schemas accordingly.
    4. Depending on the amount of data you provide (number and length of input_schemas), and its variability (affecting vocabulary size),
       you may want to tune transformer parameters such as n_embd, n_head, n_layer, etc.,
       which contribute to the model's ability to learn from data, in order to optimize model performance.
       (if you want to learn more about these transformer parameters, please refer to external sources).


  Data can consist of:
    - Historical price data of any traded financial asset, such as stocks, bonds, ETFs, forex, commodities, cryptocurrencies, derivatives, ...),
    - Statistical data analysis and technical indicators, such as percentage changes, moving averages, Bollinger Bands, MACD, VWAP, RSI, ...
    - Trade-related metrics, such as volume, VIX, open interest, ....
    - Date and time intervals coinciding with any of the parameters used above that could add context to the data, enhancing its training,
      such as time, day of week, day of month, ... (these can include non-numerical values as in days of week).


  You can choose to apply one or more of the available built in functionality for processing the data,
  such as percent changes, price ranging, and k-means clustering.
  Data quality- make sure the data is 'clean', meaning no missing or erroneous elements.


  Examples of input data sets you can implement:
    1. S&P500 1-day close prices, pct change, 50-day moving avg, 200-day moving avg, upper Bollinger Band, lower Bollinger Band
    2. MSFT 10-min open, high, low, close, volume, VIX, time, day of week, day of month
    3. Closing prices of stocks belonging to the semiconductor sector
    4. Oil ETF (USO), U.S. Dollar Index (USDX) futures
    5. Pairs trading: Coca-Cola (KO), PepsiCo (PEP)


  input_schema Elements:

  Loading Parameters:
  The following elements are required for loading data:
    1. Path to a data file or to a folder containing data files to upload. Files must be of file extensions .csv or .txt.
       If a file is specified (its extension must be included in the specified file name), then only that file will be uploaded.
       If a folder is specified, then all files with the specified file extensions within that folder will be uploaded.
       So for example, if a folder contains the files:
       "AAPL.csv", "AMZN.csv", "GOOGL.csv", "META.csv", "MSFT.csv", "NVDA.csv", "TSLA.csv",
       then all 7 files will be uploaded (magnificently).
       CONCATENATED INTO ONE DATA SET AND TREATED AS A SINGLE input_schema
       (while keeping track of the individual data files in order to later create sequence batches that do not cross from one data set to the next)
    2. The column number containing the data to upload.
       If a folder was specified in the prev element, then the same column number will be applied to all files in the folder.
    3. Does the data column have a header (True/False).
       If header is set to True, when reading the column, the first element of the column will be skipped, so make sure your header is only one row.
       If a folder was specified in the first element, then the header setting will apply to all files.

  Processing Parameters:
  The following elements are optional and are intended for processing the raw data:
  [ranging values and specifying the number of decimal places allow for control over the number of unique elements, and therefore control of the vocabulary size]
    4. Convert numeric data to percent changes (bool)
    5. Number of whole digits are ______. This should be applied to numeric data only, such as prices.
       So for example:
          num_whole_digits = 1 will scale prices to a range of 1.00 to 9.99
          num_whole_digits = 2 will scale prices to a range of 10.00 to 99.99
          num_whole_digits = 3 will scale prices to a range of 100.00 to 999.99
          and so on... (the number of decimal places is specified in the next element)
    6. Number of dec places should be applied to numeric data only, such as prices
    SINCE SPECIFYING number of whole digits AND number of dec places IS OPTIONAL, YOU CAN CHOOSE NOT TO SET THEM.
    NOTE: If you opt to specify only one of either number of whole digits or number of dec places, the system will assume number of whole digits is the one specified.
    IF YOUR INTENTION IS TO SPECIFY only number of dec places, THEN YOU MUST SET BOTH (IN ORDER FOR THE SYS TO KNOW WHICH IS WHICH), AND in this case YOU CAN SET number of whole digits TO ZERO FOR IT NOT TO BE APPLIED.
    7. Bin numeric data (int)
    8. Randomness size (int between 1-3, or None)
    9. Cross-attention status (bool)
    10. Modality name (str)
    ###################################################################################
    ############  ADD DESCRIPTION FOR rand size (IT SHOULD BE BETWEEN 1-3, OR None), cross-attention STATUS, convert to percents STATUS
    ############  GET RID OF THE WORDS "optional" AND "required"

  So, each input_schema would contain:
  input_schema_1 = [0: 'path to data file or folder'(str), 1: data column number(int), 2: column has header?(bool),
                   3: convert to percentages(bool), 4: number of whole digits(int), 5: number of dec places(int),
                   6: bin data(int), 7: randomness size(int), 8: cross-attention status(bool), 9: modality name(str)]

  Example:
  input_schema_1 = ['./data_1/tick_10m/', 5, True, False, 2, 2, None, 2, True, 'S&P 500 15 min close values']
  input_schema_2 = ['./data_1/tick_10m/', 5, True, False, None, None, None, None, False, 'Candle time']
  input_schema_3 = []
  input_schema_4 = []
  input_schema_5 = []
  input_schema_6 = []
  input_schema_7 = []
  input_schema_8 = []
  input_schema_9 = []
  input_schema_10 = []


  Considerations:

  1- Synchronized dates and times:
  If your different inputs are meant to be trained together at similar dates and times,
  then make sure all relevant data is synchronized to the same dates and times in the files they're uploaded from.

  2- Matching data lengths:
  Different inputs should be of the same length in order to train together along the entire set.
  !!!!!!!! THINK WHAT TO DO IF THEY ARE OF DIFFERENT LENGTHS  --> ALERT THE USER, CUT THE LONGER DATA TO MATCH THAT OF THE SHORTER (NOT IDEAL), OR EXTEND THE SHORT ONE (MAYBE DO THIS ONE)
  !!!!!!!! MAYBE DO BOTH THE 1ST AND LAST OPTIONS ABOVE

 Use as many input_schema lists as you need, in succession, starting from the 1st.
 You can leave unused lists as empty lists, or you can add additional ones in addition to the ones provided here below.
 num_input_schemas must match the number of input_schema lists below that are in use (exclude empty lists)

'''

# Number of input schemas available for programmatic configuration
num_input_schemas = 10



# input_schema_n is a list containing 10 elements:
    # 1. Path to a data file or a folder containing data files. Files must have '.csv' or '.txt' extensions (str).
    # 2. Column number - The 1-based index of the column to extract data from (int).
    # 3. Header - Boolean indicating if the data column has a header row (bool).
    # 4. Percent changes - Boolean indicating if the data should be converted to percentage changes (bool or None).
    # 5. Range - Number of whole digits, used for ranging (int or None).
    # 6. Decimal places - Number of decimal places (int or None).
    # 7. Bins - Number bins, used for binning data (int or None).
    # 8. Randomness size, used for data augmentation (int or None).
    # 9. Cross-attention status, used for model configuration (bool or None).
    # 10. Modality name (str or None).

    # Elements:  [Path, Col Num, Header, Percent Changes, Num Whole Digits, Decimal Places, Bins, Rand Size, Cross-Attend, Modality Name]
    # Types:     [(str), (int), (bool), (bool or None), (int or None), (int or None), (int or None), (int or None), (bool or None), (str or None)]
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