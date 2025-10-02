# Demo Example

This is a minimal demonstration using a small synthetic dataset (100 rows) to illustrate system basics.

## Purpose

This demo exists to:
- Demonstrate basic configuration structure
- Provide a quick "hello world" style test
- Illustrate the multimodal learning workflow

**Note**: Small datasets produce limited results. For meaningful predictions, use 100k+ rows (see main README.md).

## Running the Demo

From the main project directory:

```bash
# Copy demo configs to main directory
cp examples/demo_config.yaml config.yaml
cp examples/demo_input_schemas.yaml input_schemas.yaml

# Run training (will complete in ~30 seconds)
python main.py
```

## Expected Results

- Training completes quickly (~30-60 seconds on CPU)
- Shows the complete training workflow
- Demonstrates evaluation metrics output

## For Production Use

For real applications with meaningful results:

1. **Prepare data**: Minimum 100,000 rows (1M+ recommended, see main README.md)
2. **Configure**: Edit `config.yaml` and `input_schemas.yaml` in project root
3. **Scale up**: Use folder loading to combine multiple data sources
4. **Optimize**: Adjust model size and training iterations for your dataset

## What This Demo Covers

✅ Configuration file structure
✅ Data loading and processing
✅ Training loop execution
✅ Multimodal architecture
✅ Evaluation metrics

## Files

- `demo_data/demo_stock.csv` - 100 rows of synthetic stock data
- `demo_config.yaml` - Minimal training configuration
- `demo_input_schemas.yaml` - Two simple modalities

---

For detailed documentation and production usage, see the main README.md.
