#!/usr/bin/env python3
"""
Debug script to find zero values in the data that are causing percentage conversion errors.
"""

import os
import pandas as pd
from pathlib import Path

def check_file_for_zeros(file_path, column_number=7):
    """Check a single file for zero values in the specified column."""
    try:
        # Try reading with different delimiters
        df = None
        for delimiter in [',', ';']:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, header=0)
                if len(df.columns) >= column_number:
                    break
            except:
                continue

        if df is None or len(df.columns) < column_number:
            print(f"Could not read {file_path} or insufficient columns")
            return []

        # Get the specified column (1-based to 0-based conversion)
        column_data = df.iloc[:, column_number - 1]

        # Find zero values
        zero_indices = []
        for i, value in enumerate(column_data):
            try:
                if float(value) == 0.0:
                    zero_indices.append(i)
            except (ValueError, TypeError):
                # Non-numeric value
                continue

        return zero_indices

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def analyze_data_folder(folder_path, column_number=7):
    """Analyze all files in the folder for zero values."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return

    print(f"Analyzing folder: {folder_path}")
    print(f"Looking for zero values in column {column_number} (Close price)")
    print("=" * 60)

    files = sorted([f for f in folder.glob("*.txt") if f.is_file()])
    total_data_points = 0
    total_zeros = 0

    for file_path in files:
        print(f"\nChecking: {file_path.name}")
        zero_indices = check_file_for_zeros(file_path, column_number)

        if zero_indices:
            print(f"  WARNING: Found {len(zero_indices)} zero values at indices: {zero_indices}")
            total_zeros += len(zero_indices)

            # Show the actual data around the zero values
            try:
                df = pd.read_csv(file_path, header=0)
                for idx in zero_indices[:3]:  # Show first 3 zeros
                    if idx > 0 and idx < len(df):
                        prev_val = df.iloc[idx-1, column_number-1]
                        zero_val = df.iloc[idx, column_number-1]
                        next_val = df.iloc[idx+1, column_number-1] if idx+1 < len(df) else "N/A"
                        print(f"    Index {idx}: {prev_val} -> {zero_val} -> {next_val}")
            except:
                pass
        else:
            print(f"  OK: No zero values found")

        # Count total data points
        try:
            df = pd.read_csv(file_path, header=0)
            total_data_points += len(df)
        except:
            pass

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"Total files analyzed: {len(files)}")
    print(f"Total data points: {total_data_points:,}")
    print(f"Total zero values found: {total_zeros}")
    print(f"Percentage of zeros: {(total_zeros/total_data_points*100):.4f}%" if total_data_points > 0 else "N/A")

def find_cumulative_index_1359():
    """Find which file and index corresponds to cumulative index 1359."""
    folder = Path("./data/1day_candles/")
    files = sorted([f for f in folder.glob("*.txt") if f.is_file()])

    cumulative_count = 0
    target_index = 1359

    print(f"Finding cumulative index {target_index}...")
    print("=" * 50)

    for file_path in files:
        try:
            df = pd.read_csv(file_path, header=0)
            file_length = len(df)

            if cumulative_count + file_length > target_index:
                # The target index is in this file
                local_index = target_index - cumulative_count
                print(f"Found at:")
                print(f"  File: {file_path.name}")
                print(f"  Local index: {local_index}")
                print(f"  Cumulative index: {target_index}")

                # Show the data around this index
                if local_index > 0:
                    prev_val = df.iloc[local_index-1, 6]  # Column 7 (0-based = 6)
                    curr_val = df.iloc[local_index, 6]
                    next_val = df.iloc[local_index+1, 6] if local_index+1 < len(df) else "N/A"

                    print(f"  Data around index {local_index}:")
                    print(f"    Index {local_index-1}: {prev_val}")
                    print(f"    Index {local_index}: {curr_val} <- Problem value")
                    print(f"    Index {local_index+1}: {next_val}")
                break

            cumulative_count += file_length
            print(f"  {file_path.name}: {file_length} rows (cumulative: {cumulative_count})")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    print("ZERO VALUE DETECTION SCRIPT")
    print("=" * 60)

    # Find the specific problematic index
    find_cumulative_index_1359()

    print("\n" + "=" * 60)

    # Analyze entire dataset
    analyze_data_folder("./data/1day_candles/", column_number=7)