"""
File caching system for multimodal data loading.

Prevents reloading the same files multiple times when different modalities
use different columns from the same source files.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob


class FileCache:
    """
    Caches loaded files to avoid repeated disk I/O when multiple modalities
    use different columns from the same source files.
    """

    def __init__(self, max_memory_mb: float = 500.0, max_files: int = 200):
        """
        Initialize file cache with limits.

        Args:
            max_memory_mb: Maximum memory usage in MB (default 500MB)
            max_files: Maximum number of cached files (default 200)
        """
        self.cache: Dict[str, pd.DataFrame] = {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_files = max_files
        self.access_order = []  # Track access order for LRU eviction
        self.load_stats = {
            'files_loaded': 0,
            'cache_hits': 0,
            'total_requests': 0,
            'evictions': 0
        }

    def get_dataframe(self, file_path: str, has_header: bool = True) -> pd.DataFrame:
        """
        Get DataFrame from cache or load from file.

        Args:
            file_path: Path to the file
            has_header: Whether the file has a header row

        Returns:
            pandas DataFrame containing the file data
        """
        # Normalize path for consistent caching
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        cache_key = f"{normalized_path}_{has_header}"

        self.load_stats['total_requests'] += 1

        if cache_key in self.cache:
            self.load_stats['cache_hits'] += 1
            # Update access order for LRU
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]

        # File not in cache, load it
        self.load_stats['files_loaded'] += 1
        df = self._load_file(file_path, has_header)

        # Check if we need to evict before adding
        self._ensure_capacity()

        self.cache[cache_key] = df
        self.access_order.append(cache_key)
        return df

    def _load_file(self, file_path: str, has_header: bool) -> pd.DataFrame:
        """
        Load a single file into a DataFrame.

        Args:
            file_path: Path to the file
            has_header: Whether the file has a header row

        Returns:
            pandas DataFrame
        """
        header = 0 if has_header else None

        # Try different delimiters
        for delimiter in [',', ';']:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, header=header)
                if len(df.columns) > 1:  # Successfully parsed multiple columns
                    return df
            except Exception:
                continue

        # Fallback: try default pandas behavior
        try:
            return pd.read_csv(file_path, header=header)
        except Exception as e:
            raise RuntimeError(f"Failed to load file {file_path}: {e}")

    def get_column_data(self, file_path: str, column_number: int, has_header: bool = True) -> List:
        """
        Get data from a specific column of a file.

        Args:
            file_path: Path to the file
            column_number: 1-based column index
            has_header: Whether the file has a header row

        Returns:
            List of values from the specified column
        """
        df = self.get_dataframe(file_path, has_header)

        # Convert to 0-based index
        col_index = column_number - 1

        if col_index >= len(df.columns):
            raise ValueError(f"Column {column_number} does not exist in file {file_path}. "
                           f"File has {len(df.columns)} columns.")

        # Return as list, handling NaN values
        column_data = df.iloc[:, col_index].tolist()
        return column_data

    def load_multiple_files(self, folder_path: str, column_number: int, has_header: bool = True) -> Tuple[List, List]:
        """
        Load data from multiple files in a folder using caching.

        Args:
            folder_path: Path to folder containing files
            column_number: 1-based column index to extract
            has_header: Whether files have header rows

        Returns:
            Tuple of (all_data, file_info)
            - all_data: Combined data from all files
            - file_info: [file1_name, file1_length, file2_name, file2_length, ...]
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path {folder_path} is not a directory")

        # Find all CSV and TXT files
        file_patterns = ['*.csv', '*.txt']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(folder_path, pattern)))

        if not all_files:
            raise ValueError(f"No CSV or TXT files found in {folder_path}")

        # Sort files for consistent ordering
        all_files.sort()

        combined_data = []
        file_info = []

        for file_path in all_files:
            file_name = os.path.basename(file_path)

            # Get column data using cache
            column_data = self.get_column_data(file_path, column_number, has_header)

            # Add to combined data
            combined_data.extend(column_data)

            # Add file info
            file_info.extend([file_name, len(column_data)])

        return combined_data, file_info

    def get_cache_stats(self) -> Dict:
        """Get caching statistics."""
        stats = self.load_stats.copy()
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_requests']) * 100
        else:
            stats['cache_hit_rate'] = 0
        stats['cached_files'] = len(self.cache)
        return stats

    def _ensure_capacity(self):
        """Ensure cache doesn't exceed memory or file limits."""
        # Check file count limit
        while len(self.cache) >= self.max_files:
            self._evict_lru()

        # Check memory limit
        current_memory = sum(df.memory_usage(deep=True).sum() for df in self.cache.values())
        while current_memory > self.max_memory_bytes and self.cache:
            self._evict_lru()
            current_memory = sum(df.memory_usage(deep=True).sum() for df in self.cache.values())

    def _evict_lru(self):
        """Evict least recently used item from cache."""
        if not self.access_order:
            return

        lru_key = self.access_order.pop(0)
        if lru_key in self.cache:
            del self.cache[lru_key]
            self.load_stats['evictions'] += 1

    def clear_cache(self):
        """Clear the file cache and reset statistics."""
        self.cache.clear()
        self.access_order.clear()
        self.load_stats = {
            'files_loaded': 0,
            'cache_hits': 0,
            'total_requests': 0,
            'evictions': 0
        }

    def get_memory_usage(self) -> Dict:
        """
        Get approximate memory usage of cached files.

        Returns:
            Dictionary with memory usage information
        """
        total_memory = 0
        file_details = {}

        for cache_key, df in self.cache.items():
            memory_bytes = df.memory_usage(deep=True).sum()
            total_memory += memory_bytes
            file_details[cache_key] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': memory_bytes / (1024 * 1024)
            }

        return {
            'total_memory_mb': total_memory / (1024 * 1024),
            'cached_files_count': len(self.cache),
            'file_details': file_details
        }


# Global cache instance
_file_cache = FileCache()


def get_file_cache() -> FileCache:
    """Get the global file cache instance."""
    return _file_cache


def load_file_data_cached(input_info: List) -> Tuple[List, List]:
    """
    Enhanced version of load_file_data that uses caching to avoid reloading same files.

    This function replaces the original load_file_data for better performance when
    multiple modalities use different columns from the same source files.

    Args:
        input_info: Same format as original load_file_data

    Returns:
        Tuple of (data, file_info) - same format as original
    """
    if not isinstance(input_info, list) or len(input_info) != 10:
        raise ValueError("'input_info' must contain 10 elements")

    data_path = input_info[0]
    column_number = input_info[1]
    has_header = input_info[2]
    convert_to_percentages = input_info[3]

    cache = get_file_cache()

    # Handle both single files and folders
    if os.path.isfile(data_path):
        # Single file
        column_data = cache.get_column_data(data_path, column_number, has_header)
        file_name = os.path.basename(data_path)
        file_info = [file_name, len(column_data)]
        all_data = column_data
    else:
        # Folder with multiple files
        all_data, file_info = cache.load_multiple_files(data_path, column_number, has_header)

    # Apply percentage conversion if requested
    if convert_to_percentages:
        all_data = _calculate_percentage_changes(all_data)

    return all_data, file_info


def _calculate_percentage_changes(data: List) -> List[float]:
    """Calculate percentage changes from a list of numeric values."""
    if not data:
        return []

    percentages = [0.0]  # First element is always 0 (no previous value)

    for i in range(1, len(data)):
        try:
            current = float(data[i])
            previous = float(data[i-1])

            if previous == 0:
                raise ZeroDivisionError(f"Cannot calculate percentage change: previous value is zero at index {i-1}")

            percentage_change = ((current - previous) / previous) * 100
            percentages.append(percentage_change)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Non-numeric data encountered at index {i}: {data[i]}. "
                           f"Cannot calculate percentage change: {e}")

    return percentages


def print_cache_stats():
    """Print caching statistics for debugging."""
    cache = get_file_cache()
    stats = cache.get_cache_stats()
    memory = cache.get_memory_usage()

    print(f"\\nFile Cache Statistics:")
    print(f"  Files loaded from disk: {stats['files_loaded']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"  Cache evictions: {stats['evictions']}")
    print(f"  Cached files: {stats['cached_files']}")
    print(f"  Memory usage: {memory['total_memory_mb']:.1f} MB")

def cleanup_cache():
    """Clear the cache to free memory after data loading is complete."""
    cache = get_file_cache()
    memory_before = cache.get_memory_usage()['total_memory_mb']
    cache.clear_cache()
    print(f"\\nCache cleaned up - released {memory_before:.1f} MB of memory")


if __name__ == "__main__":
    # Example usage
    cache = get_file_cache()

    # Simulate loading same file multiple times
    print("Demo: File caching system")

    # This would normally load the file from disk multiple times
    # With caching, only the first load hits the disk
    test_input = ["./data/test.csv", 1, True, False, None, None, None, None, None, "Test"]

    print("Note: Run this with actual data files to see caching in action")
    print_cache_stats()