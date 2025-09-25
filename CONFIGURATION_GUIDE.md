# Trade-AId Configuration Guide

This guide explains how to configure the Trade-AId multimodal transformer system for your specific use case.

## Configuration Methods

The system supports two configuration approaches:

### 🚀 **YAML Configuration (Recommended)**
- **Files**: `config.yaml` + `input_schemas.yaml`
- **Best for**: Most users, clean separation of concerns
- **Benefits**: User-friendly, well-documented, validation built-in

### 🔧 **Programmatic Configuration**
- **Files**: `config.py` with input_schema definitions
- **Best for**: Advanced users, automation, dynamic configuration
- **Benefits**: Full Python flexibility, programmatic control

The system automatically detects which method you're using and adapts accordingly.

---

## Input Data Configuration

### What is an Input Schema?

An **input schema** defines one **modality** - a specific data source that the model will learn from. Each modality represents a different aspect of your data (e.g., stock prices, volume, technical indicators, time features).

### Multimodal Learning

The system is designed to find connections and patterns **between** different modalities using selective cross-attention. For example:
- **Modality 1**: S&P 500 closing prices
- **Modality 2**: Trading volume
- **Modality 3**: VIX volatility index
- **Modality 4**: Day of week

The model learns how these different data sources influence each other.

---

## Data Requirements

### File Formats
- **Supported**: `.csv` and `.txt` files
- **Single file**: Specify exact filename (e.g., `"./data/AAPL.csv"`)
- **Multiple files**: Specify folder path (e.g., `"./data/stocks/"`)

When using a folder, all `.csv` and `.txt` files are:
1. Loaded automatically
2. Concatenated into one dataset
3. Treated as a single modality
4. File boundaries preserved for proper sequence batching

### Data Quality
- **Clean data**: No missing or erroneous values
- **Synchronized timing**: All modalities should align temporally
- **Consistent length**: All modalities should have the same number of data points

---

## Input Schema Elements

Each input schema contains 10 elements:

| Index | Parameter | Type | Description |
|-------|-----------|------|-------------|
| 0 | **Path** | `str` | File path or folder path |
| 1 | **Column Number** | `int` | 1-based column index to extract |
| 2 | **Has Header** | `bool` | Whether first row is a header |
| 3 | **Convert to Percentages** | `bool` | Calculate percentage changes |
| 4 | **Whole Digits** | `int/None` | Number of digits for ranging |
| 5 | **Decimal Places** | `int/None` | Decimal precision for ranging |
| 6 | **Bins** | `int/None` | Number of bins for binning |
| 7 | **Randomness Size** | `int/None` | Data augmentation (1-3) |
| 8 | **Cross-Attention** | `bool` | Enable cross-modal attention |
| 9 | **Modality Name** | `str` | Descriptive name |

---

## Data Processing Options

### 1. Percentage Changes (`convert_to_percentages`)
```python
# Convert raw prices to percentage changes
# 100.0, 102.0, 101.0 → 0%, 2%, -0.98%
convert_to_percentages = True
```

### 2. Ranging (`num_whole_digits`, `decimal_places`)
Controls vocabulary size by standardizing numeric precision:

```python
# Examples:
num_whole_digits = 2, decimal_places = 1  # Range: 10.0 to 99.9
num_whole_digits = 3, decimal_places = 2  # Range: 100.00 to 999.99
```

**Special cases**:
- Only specify `num_whole_digits`: System assumes that's what you want
- Specify only `decimal_places`: Set `num_whole_digits = 0`

### 3. Binning (`num_bins`)
```python
# Group continuous values into discrete bins
num_bins = 50  # Creates 50 evenly-distributed bins
```

### 4. Randomness (`randomness_size`)
```python
# Add noise for data augmentation (1-3 scale)
randomness_size = 2  # Moderate randomization
```

### 5. Cross-Attention (`cross_attention`)
```python
# Allow this modality to attend to others
cross_attention = True   # This modality can see other modalities
cross_attention = False  # This modality is independent
```

---

## Example Configurations

### Basic Stock Analysis
```python
input_schema_1 = [
    './data/stocks/',           # Folder with multiple stock files
    4,                          # Column 4 (closing price)
    True,                       # Has header row
    False,                      # Don't convert to percentages
    2, 1,                       # Range to 2 digits, 1 decimal (10.0-99.9)
    None,                       # No binning
    None,                       # No randomness
    True,                       # Enable cross-attention
    'Stock Prices'              # Modality name
]

input_schema_2 = [
    './data/stocks/',           # Same folder
    7,                          # Column 7 (volume)
    True,                       # Has header row
    True,                       # Convert to percentage changes
    None, 2,                    # Only decimal places
    10,                         # Bin into 10 groups
    1,                          # Light randomization
    True,                       # Enable cross-attention
    'Volume Changes'            # Modality name
]
```

### Multi-Asset Strategy
```python
# Gold prices
input_schema_1 = ['./data/gold.csv', 2, True, False, 4, 2, None, None, True, 'Gold']

# Oil futures
input_schema_2 = ['./data/oil.csv', 3, True, True, None, 1, 5, None, True, 'Oil Returns']

# USD Index
input_schema_3 = ['./data/usd.csv', 2, True, False, 2, 3, None, 2, True, 'USD Index']

# Market timing (day of week)
input_schema_4 = ['./data/timing.csv', 1, True, False, None, None, None, None, False, 'Day of Week']
```

---

## Data Examples

### Suitable Data Types

**Financial Time Series**:
- Stock prices (OHLCV)
- Forex rates
- Cryptocurrency prices
- Bond yields
- Commodity prices

**Technical Indicators**:
- Moving averages (SMA, EMA)
- Oscillators (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume indicators (VWAP, OBV)

**Market Data**:
- VIX (volatility index)
- Interest rates
- Economic indicators
- Sector performance

**Contextual Features**:
- Time (hour, day, month)
- Market session (pre-market, regular, after-hours)
- Calendar events (earnings, Fed meetings)
- Seasonal factors

---

## Best Practices

### 1. **Data Synchronization**
```python
# ✅ Good: All data points align temporally
timestamp_1 = ['2023-01-01 09:30', '2023-01-01 09:31', ...]  # Stock prices
timestamp_2 = ['2023-01-01 09:30', '2023-01-01 09:31', ...]  # Volume

# ❌ Bad: Different timestamps
timestamp_1 = ['2023-01-01 09:30', '2023-01-01 09:31', ...]  # Stock prices
timestamp_2 = ['2023-01-01 10:00', '2023-01-01 10:01', ...]  # Volume
```

### 2. **Vocabulary Size Control**
- **Large vocabularies** = More unique values = Harder to learn
- **Small vocabularies** = Less precision = Less information
- Use **ranging** and **binning** to find the right balance

### 3. **Cross-Attention Strategy**
```python
# Primary signal modalities: Enable cross-attention
price_modality = [..., True, 'Prices']      # Can see other modalities
volume_modality = [..., True, 'Volume']     # Can see other modalities

# Context modalities: Often don't need cross-attention
time_modality = [..., False, 'Time']        # Independent context
```

### 4. **Processing Pipeline Order**
The system processes data in this order:
1. **Percentage changes** (if enabled)
2. **Ranging** (if specified)
3. **Binning** (if specified)

Plan your processing chain accordingly.

---

## Troubleshooting

### Common Issues

**"Modalities have different lengths"**
- Ensure all data files have the same number of rows
- Check for missing data or different time ranges

**"Vocabulary too large"**
- Use `ranging` to reduce precision
- Use `binning` to group similar values
- Consider percentage changes for price data

**"Model not learning"**
- Check data quality and synchronization
- Verify cross-attention settings
- Consider adjusting processing parameters

**"Training too slow"**
- Reduce vocabulary sizes (ranging/binning)
- Decrease `block_size` or `batch_size`
- Use fewer modalities initially

---

## Advanced Configuration

### Custom Processing Functions
When using YAML configuration, you can define custom processing functions in the `processing_steps` section. See `input_schemas.yaml` for examples.

### Hyperparameter Tuning
The model architecture parameters (`n_embd`, `n_head`, `n_layer`) should be adjusted based on:
- **Data complexity**: More complex relationships need larger models
- **Vocabulary sizes**: Larger vocabularies need more embedding dimensions
- **Available compute**: Larger models need more GPU memory

### Evaluation Parameters
Understanding the difference between evaluation frequency and thoroughness:

- **`eval_interval`** (Evaluation Frequency): Controls **when** evaluation happens
  - Sets how often to run validation during training (every N iterations)
  - Example: `eval_interval: 50` = evaluate every 50 training steps
  - More frequent = better monitoring, but slower training

- **`eval_iters`** (Evaluation Thoroughness): Controls **how thorough** each evaluation is
  - Sets how many validation batches to use per evaluation
  - Example: `eval_iters: 40` = use 40 batches to calculate validation loss
  - More batches = more accurate loss estimates, but slower per evaluation

### Performance Optimization
- **File caching**: System automatically caches loaded files for better performance
- **Lazy evaluation**: Configuration values are loaded only when needed
- **Memory management**: Cache is automatically cleared after data loading

---

## Getting Help

- Check the console output for detailed error messages
- Verify your data files are accessible and properly formatted
- Start with simple configurations and gradually add complexity
- Use small datasets for initial testing and validation

For more advanced configuration options, see the YAML configuration files (`config.yaml` and `input_schemas.yaml`) for comprehensive examples and documentation.