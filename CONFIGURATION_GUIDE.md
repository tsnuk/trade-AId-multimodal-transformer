# Trade-AId Configuration Guide

Quick reference for configuring the multimodal transformer system. For detailed explanations and tutorials, see [README.md](README.md).

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

```yaml
# Examples:
num_whole_digits: 2
decimal_places: 1  # Range: 10.0 to 99.9

num_whole_digits: 3
decimal_places: 2  # Range: 100.00 to 999.99

# Skip processing entirely
num_whole_digits: null
decimal_places: null
```

**Special cases**:
- Only specify `num_whole_digits`: System assumes that's what you want
- Specify only `decimal_places`: Set `num_whole_digits: null`
- Both `null`: No ranging applied

### 3. Binning (`num_bins`)
```python
# Group continuous values into discrete bins
num_bins = 50  # Creates 50 evenly-distributed bins
```

### 4. Randomness (`randomness_size`)
```yaml
# Add noise for data augmentation (1-3 or null)
randomness_size: 2  # Moderate randomization, or null for none
```

### 5. Cross-Attention (`cross_attention`)
```python
# Allow this modality to attend to others
cross_attention = True   # This modality can see other modalities
cross_attention = False  # This modality is independent
```

---

## Quick Examples

For detailed examples and tutorials, see [README.md](README.md#example-modality-configurations).

### Basic Schema Format
```python
input_schema_1 = [path, column, header, percentages, digits, decimals, bins, randomness, cross_attn, name]
```

### Common Data Types
- **Financial**: Stock prices, forex, crypto, bonds, commodities
- **Indicators**: RSI, MACD, moving averages, Bollinger Bands
- **Market**: VIX, volume, interest rates, sector performance
- **Context**: Time, day of week, market sessions, events

---

## Best Practices

- **Data Alignment**: All modalities must have same-length, temporally synchronized datasets
- **Vocabulary Control**: Use ranging/binning to balance precision vs. learning efficiency
- **Cross-Attention**: Enable for related data, disable for independent context (time, categorical)
- **Processing Order**: Percentages → Ranging → Binning

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