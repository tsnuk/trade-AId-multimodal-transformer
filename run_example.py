#!/usr/bin/env python3
"""
Quick Start Script for Multimodal Transformer Examples

This script provides an easy way to run the included examples without
manually copying configuration files.

Usage:
    python run_example.py 1    # Run Basic Example
    python run_example.py 2    # Run Advanced Example
    python run_example.py --list    # List available examples
"""

import sys
import shutil
from pathlib import Path

def print_banner():
    """Print a nice banner for the examples"""
    print("=" * 70)
    print("🚀 MULTIMODAL TRANSFORMER EXAMPLES")
    print("=" * 70)
    print()

def print_example_info(example_num):
    """Print information about the selected example"""
    if example_num == 1:
        print("📚 EXAMPLE 1: BASIC MULTIMODAL LEARNING")
        print("   • Perfect for beginners and quick testing")
        print("   • 4 modalities from single data file")
        print("   • Small model (32 dim, 2 layers)")
        print("   • CPU-optimized, 100 iterations")
        print("   • Estimated time: 1-2 minutes")
        print()
    elif example_num == 2:
        print("⚡ EXAMPLE 2: ADVANCED MULTIMODAL LEARNING")
        print("   • Advanced features and production patterns")
        print("   • 4 modalities from multiple data files")
        print("   • Larger model (128 dim, 6 layers)")
        print("   • GPU-optimized, 500 iterations")
        print("   • Estimated time: 5-10 minutes")
        print()

def list_examples():
    """List all available examples"""
    print_banner()
    print("📋 AVAILABLE EXAMPLES:")
    print()
    print("1️⃣  Basic Multimodal Learning")
    print("    → python run_example.py 1")
    print("    → Beginner-friendly, single file, CPU-optimized")
    print()
    print("2️⃣  Advanced Multimodal Learning")
    print("    → python run_example.py 2")
    print("    → Multiple files, complex processing, GPU-optimized")
    print()
    print("For more details, see examples/README.md")
    print()

def backup_existing_configs():
    """Backup existing configuration files if they exist"""
    backups_made = []

    if Path("config.yaml").exists():
        shutil.copy2("config.yaml", "config.yaml.backup")
        backups_made.append("config.yaml")

    if Path("input_schemas.yaml").exists():
        shutil.copy2("input_schemas.yaml", "input_schemas.yaml.backup")
        backups_made.append("input_schemas.yaml")

    if backups_made:
        print(f"🔄 Backed up existing configs: {', '.join(backups_made)}")
        print("   (You can restore them later from .backup files)")
        print()

    return backups_made

def copy_example_configs(example_num):
    """Copy the selected example configurations to the main directory"""
    examples_dir = Path("examples/configs")

    if not examples_dir.exists():
        print("❌ ERROR: Examples directory not found!")
        print("   Make sure you're running this script from the main project directory.")
        return False

    # Define source files based on example number
    if example_num == 1:
        config_src = examples_dir / "example1_basic_config.yaml"
        schemas_src = examples_dir / "example1_basic_input_schemas.yaml"
    elif example_num == 2:
        config_src = examples_dir / "example2_advanced_config.yaml"
        schemas_src = examples_dir / "example2_advanced_input_schemas.yaml"
    else:
        print(f"❌ ERROR: Invalid example number: {example_num}")
        print("   Available examples: 1, 2")
        return False

    # Check if source files exist
    if not config_src.exists() or not schemas_src.exists():
        print(f"❌ ERROR: Example {example_num} configuration files not found!")
        print(f"   Looking for: {config_src} and {schemas_src}")
        return False

    # Copy files
    try:
        shutil.copy2(config_src, "config.yaml")
        shutil.copy2(schemas_src, "input_schemas.yaml")
        print(f"✅ Copied Example {example_num} configurations")
        print("   → config.yaml")
        print("   → input_schemas.yaml")
        print()
        return True
    except Exception as e:
        print(f"❌ ERROR copying files: {e}")
        return False

def check_data_files(example_num):
    """Check if required data files exist"""
    if example_num == 1:
        data_file = Path("examples/sample_data/basic_multimodal/sample_stock_data.csv")
        if not data_file.exists():
            print(f"❌ ERROR: Required data file not found: {data_file}")
            return False
    elif example_num == 2:
        data_files = [
            Path("examples/sample_data/advanced_multimodal/stock_AAPL.csv"),
            Path("examples/sample_data/advanced_multimodal/stock_MSFT.csv"),
            Path("examples/sample_data/advanced_multimodal/stock_GOOGL.csv")
        ]
        missing_files = [f for f in data_files if not f.exists()]
        if missing_files:
            print("❌ ERROR: Required data files not found:")
            for f in missing_files:
                print(f"   → {f}")
            return False

    print("✅ All required data files found")
    return True

def run_training():
    """Run the main training script"""
    print("🎯 STARTING TRAINING...")
    print("=" * 50)
    print()

    # Import and run main training
    try:
        import main
        print()
        print("🎉 TRAINING COMPLETED!")
        print()
        print("📊 Check the output files for results:")
        print("   → Training logs in output/ directory")
        print("   → Model saved as .pth file")
        print()
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   → Check examples/README.md for common issues")
        print("   → Verify your Python environment has required packages")
        print("   → Try reducing batch_size or block_size if out of memory")
        return False

    return True

def main():
    """Main script entry point"""
    if len(sys.argv) != 2:
        print("Usage: python run_example.py [1|2|--list]")
        print("  1     - Run Basic Example")
        print("  2     - Run Advanced Example")
        print("  --list - List available examples")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list" or arg == "-l":
        list_examples()
        return

    try:
        example_num = int(arg)
    except ValueError:
        print(f"❌ ERROR: Invalid argument '{arg}'. Use 1, 2, or --list")
        sys.exit(1)

    if example_num not in [1, 2]:
        print(f"❌ ERROR: Invalid example number: {example_num}")
        print("   Available examples: 1, 2")
        sys.exit(1)

    # Run the example
    print_banner()
    print_example_info(example_num)

    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ ERROR: main.py not found!")
        print("   Make sure you're running this script from the main project directory.")
        sys.exit(1)

    # Check data files
    if not check_data_files(example_num):
        print("   Make sure the examples package is complete.")
        sys.exit(1)

    # Backup existing configs
    backup_existing_configs()

    # Copy example configurations
    if not copy_example_configs(example_num):
        sys.exit(1)

    # Ask for confirmation
    print("🚀 Ready to start training!")
    response = input("   Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        sys.exit(0)

    print()

    # Run training
    success = run_training()

    if success:
        print("✨ Example completed successfully!")
        print("   See examples/README.md for next steps and customization ideas.")
    else:
        print("❌ Example encountered errors.")
        print("   Check examples/README.md for troubleshooting tips.")

if __name__ == "__main__":
    main()