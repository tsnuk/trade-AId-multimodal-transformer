# Multimodal Transformer Examples

Welcome to the example configurations for the Multimodal Transformer system! These examples are designed to help you understand and experiment with multimodal learning using realistic financial data.

## Overview

This examples package provides two complete, working configurations that demonstrate different aspects of the multimodal transformer system:

- **Example 1: Basic Multimodal** - Perfect for beginners and quick experimentation
- **Example 2: Advanced Multimodal** - Demonstrates sophisticated features and production-ready patterns

Both examples use **4 modalities** to showcase the power of multimodal learning, where the model learns relationships between different types of time-series data simultaneously.

## Prerequisites

Before running the examples, ensure you have the required dependencies installed:

```bash
pip install torch pyyaml numpy pandas
```

**Note on Training Time**: Transformer training requires substantial computational resources. Depending on your hardware (CPU vs GPU) and the example's complexity, training sessions may take anywhere from a few minutes to over an hour. Example 1 typically completes in 1-2 minutes on modern hardware, while Example 2 may require 10-30 minutes or more on CPU. For faster training, consider using a CUDA-enabled GPU if available.

## Quick Start

### Option 1: Run Examples Directly

```bash
# Run Example 1 (Basic)
python run_example.py 1

# Run Example 2 (Advanced)
python run_example.py 2
```

### Option 2: Copy Configurations to Main Directory

```bash
# Copy Example 1 configurations
cp examples/configs/example1_basic_config.yaml config.yaml
cp examples/configs/example1_basic_input_schemas.yaml input_schemas.yaml
python main.py

# Or copy Example 2 configurations
cp examples/configs/example2_advanced_config.yaml config.yaml
cp examples/configs/example2_advanced_input_schemas.yaml input_schemas.yaml
python main.py
```

## Example Details

### Example 1: Basic Multimodal Learning

**Perfect for: Beginners, Learning, Quick Testing**

**Configuration:**
- **4 Modalities** from a single data file
- **Small model** (32 embedding dim, 2 layers) for fast training
- **CPU-optimized** settings
- **100 training iterations** (completes in 1-2 minutes)

**Modalities Demonstrated:**
1. **Stock Prices** - Scaled close prices with cross-attention
2. **Price Changes** - Percentage changes binned into 5 categories
3. **Hour of Day** - Time-based patterns (9-16 market hours)
4. **Day of Week** - Weekday seasonality patterns (1-7)

**Key Learning Points:**
- Single data source, multiple modalities
- Basic processing pipelines
- Cross-attention vs. independent modalities
- Time-aligned multimodal data

**Expected Results:**
- Quick training convergence
- Clear vocabulary sizes per modality
- Demonstrable cross-modal attention patterns

---

### Example 2: Advanced Multimodal Learning

**Perfect for: Advanced Users, Production Patterns, Complex Relationships**

**Configuration:**
- **4 Modalities** from multiple data files (folder loading)
- **Larger model** (128 embedding dim, 6 layers) for complex learning
- **GPU-optimized** settings with auto device detection
- **500 training iterations** with sophisticated processing

**Modalities Demonstrated:**
1. **Multi-Stock Prices** - Combined AAPL, MSFT, GOOGL with data augmentation
2. **Volatility Patterns** - Multi-step processing (percentages -> advanced binning)
3. **Trading Volume** - Scaled and categorized volume patterns
4. **Market Timing** - Time-based features (categorical hour data)

**Advanced Features:**
- **Folder-based loading** - Multiple files per modality
- **Multi-step processing pipelines** - Complex data transformations
- **File-based validation** - Last file reserved for testing generalization
- **Data augmentation** - Strategic randomness injection during training
- **Strategic cross-attention** - Optimized attention patterns

**Expected Results:**
- Sophisticated multimodal interactions
- Better generalization across different stocks
- Complex pattern recognition capabilities

## Sample Data

### Data Format
All sample data follows the same format as real financial data:
```csv
datetime,year,month,day,day_of_week,hour,minute,date,time,open,high,low,close,volume
2024-01-02 09:30:00-05:00,2024,1,2,2,9,30,2024-01-02,09:30:00,180.50,181.20,179.80,180.90,1250000.0
```

### Data Sources
- **Example 1**: Single file with 50 time periods (sample_stock_data.csv)
- **Example 2**: Three separate stock files (AAPL, MSFT, GOOGL) with 40+ time periods each

### Data Characteristics
- **Realistic price movements** - Based on actual market patterns
- **Appropriate volume ranges** - Scaled for different stock sizes
- **Time-aligned data** - Perfect synchronization across all columns
- **Clean formatting** - No missing values or inconsistencies

## Customization Guide

### Modifying Examples

1. **Change Data Sources**:
   ```yaml
   # In input_schemas.yaml
   - modality_name: "Your Data"
     path: "./your_data_folder/"
     column_number: 5  # Your column of interest
   ```

2. **Adjust Model Size**:
   ```yaml
   # In config.yaml
   model_architecture:
     n_embd: 64      # Larger = more complex
     n_head: 8       # Must divide n_embd evenly
     n_layer: 4      # More layers = deeper learning
   ```

3. **Modify Processing**:
   ```yaml
   # In input_schemas.yaml
   processing_steps:
     - function: your_custom_function
       args: {param1: value1}
     - function: built_in_function
       args: {param2: value2}
   ```

### Creating Your Own Examples

1. **Copy an existing example**
2. **Modify data paths** to point to your data
3. **Adjust column numbers** for your CSV structure
4. **Update processing steps** for your data types
5. **Tune model parameters** for your dataset size

## Expected Outputs

### Training Logs
Both examples generate detailed training logs showing:
- **Hyperparameters** and model architecture
- **Data statistics** (sizes, vocabularies, splits)
- **Training progress** (loss, evaluation metrics)
- **Multimodal evaluation** (per-modality and combined metrics)
- **Directional success rates** (financial prediction accuracy)

### Model Files
- Trained model weights saved as `.pth` files
- Can be loaded for further training or inference
- Include complete model state and optimizer information

### Performance Metrics
- **Overall Loss** - Combined multimodal training loss
- **Directional Success** - Per-modality directional prediction accuracy (up/down/flat)

## Troubleshooting

### Common Issues

**"Path does not exist"**
- Check that example data files are present
- Verify paths in configuration files are correct
- Ensure you're running from the main project directory

**"CUDA out of memory"**
- Reduce `batch_size` in config.yaml
- Reduce `block_size` for shorter sequences
- Switch to `device: cpu` in config.yaml

**"Modality data lengths don't match"**
- Ensure all data files have the same number of rows
- Check that processing steps don't filter different amounts of data
- Verify file formats are consistent

**Training is very slow**
- Use Example 1 for faster training
- Reduce `max_iters` for quicker results
- Switch to `device: cpu` if GPU is slower

### Getting Help

1. **Check the main README.md** for detailed documentation
2. **Review configuration comments** - All examples are heavily commented
3. **Start with Example 1** - Simpler setup for debugging
4. **Compare with working examples** - Use examples as reference templates

## Learning Path

### Recommended Progression

1. **Start with Example 1**
   - Understand basic multimodal concepts
   - See how different modalities interact
   - Experiment with small changes

2. **Move to Example 2**
   - Learn advanced processing pipelines
   - Understand folder-based loading
   - Explore cross-attention strategies

3. **Create Your Own**
   - Use your own data sources
   - Design custom processing pipelines
   - Optimize for your specific use case

### Key Concepts to Master

- **Multimodal Data Alignment** - How different data types synchronize
- **Processing Pipelines** - Transforming raw data for neural networks
- **Cross-Attention Patterns** - When modalities should interact
- **Data Augmentation** - Using randomness for better generalization
- **Validation Strategies** - Percentage vs. file-based splitting

## Next Steps

After working with these examples:

1. **Try your own financial data** - Stock prices, crypto, forex
2. **Experiment with different modalities** - News sentiment, economic indicators
3. **Design custom processing functions** - Create your own transformations
4. **Scale up model size** - For larger datasets and complex patterns
5. **Explore advanced features** - Custom attention patterns, external functions

Happy learning!