"""processing_registry.py

Function registry for built-in data processing functions.

This module maintains a dictionary mapping simple string names to actual Python
function objects for built-in processing functions. It enables the processing
pipeline to resolve both built-in functions (by simple name) and external
functions (by fully qualified module.function name).

Usage:
    # Built-in function lookup
    func = builtin_processing_functions['range_numeric_data']

    # External function resolution handled by processing pipeline
    func = resolve_function('my_module.custom_function')
"""

import importlib
from typing import Callable, Dict, Any, List

from data_utils import (
    range_numeric_data,
    bin_numeric_data,
    convert_to_percent_changes,
    add_rand_to_data_points
)

builtin_processing_functions: Dict[str, Callable] = {
    'range_numeric_data': range_numeric_data,
    'bin_numeric_data': bin_numeric_data,
    'convert_to_percent_changes': convert_to_percent_changes,
    'add_rand_to_data_points': add_rand_to_data_points,
}


def resolve_function(function_name: str) -> Callable:
    """
    Resolve a function by name, checking built-in registry first, then attempting
    dynamic import for external functions.

    Args:
        function_name: Either a simple name (for built-ins) or fully qualified
                      name like 'module.submodule.function_name' (for external)

    Returns:
        The resolved function object

    Raises:
        ImportError: If the function cannot be found or imported
        AttributeError: If the function doesn't exist in the specified module
        ValueError: If the function name is invalid
    """
    if not function_name or not isinstance(function_name, str):
        raise ValueError(f"Function name must be a non-empty string, got: {function_name}")

    if function_name in builtin_processing_functions:
        return builtin_processing_functions[function_name]

    try:
        if '.' not in function_name:
            raise ImportError(f"External function '{function_name}' must be fully qualified (e.g., 'module.function')")

        module_name, func_name = function_name.rsplit('.', 1)

        module = importlib.import_module(module_name)

        if not hasattr(module, func_name):
            raise AttributeError(f"Module '{module_name}' has no function '{func_name}'")

        function_obj = getattr(module, func_name)

        if not callable(function_obj):
            raise TypeError(f"'{function_name}' is not a callable function")

        return function_obj

    except ImportError as e:
        raise ImportError(f"Failed to import external function '{function_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Failed to resolve external function '{function_name}': {e}")
    except Exception as e:
        raise ImportError(f"Unexpected error resolving function '{function_name}': {e}")


def get_available_builtin_functions() -> List[str]:
    """Get a list of all available built-in function names.

    Returns:
        List of built-in function names that can be used in configuration.
    """
    return list(builtin_processing_functions.keys())


def validate_function_exists(function_name: str) -> bool:
    """Check if a function exists without actually importing it.

    Args:
        function_name: Function name to validate.

    Returns:
        True if function exists and is callable, False otherwise.
    """
    try:
        resolve_function(function_name)
        return True
    except (ImportError, AttributeError, ValueError, TypeError):
        return False


def register_builtin_function(name: str, function: Callable) -> None:
    """Register a new built-in function in the registry.

    Args:
        name: Simple string name to use in configurations.
        function: The actual function object.

    Raises:
        ValueError: If name is invalid or function is not callable.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Function name must be a non-empty string")

    if not callable(function):
        raise ValueError("Function must be callable")

    if name in builtin_processing_functions:
        print(f"Warning: Overwriting existing built-in function '{name}'")

    builtin_processing_functions[name] = function


def unregister_builtin_function(name: str) -> bool:
    """Remove a built-in function from the registry.

    Args:
        name: Function name to remove.

    Returns:
        True if function was removed, False if it didn't exist.
    """
    if name in builtin_processing_functions:
        del builtin_processing_functions[name]
        return True
    return False


BUILTIN_FUNCTION_VALIDATION = {
    'range_numeric_data': {
        'required': [],
        'optional': ['num_whole_digits', 'decimal_places'],
        'types': {
            'num_whole_digits': (int, type(None)),
            'decimal_places': (int, type(None))
        },
        'validators': {
            'num_whole_digits': lambda x: x is None or (isinstance(x, int) and x > 0),
            'decimal_places': lambda x: x is None or (isinstance(x, int) and x >= 0)
        }
    },
    'bin_numeric_data': {
        'required': ['num_bins'],
        'optional': ['outlier_percentile', 'exponent'],
        'types': {
            'num_bins': int,
            'outlier_percentile': (int, float),
            'exponent': (int, float)
        },
        'validators': {
            'num_bins': lambda x: isinstance(x, int) and x > 0,
            'outlier_percentile': lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
            'exponent': lambda x: isinstance(x, (int, float)) and x > 0
        }
    },
    'convert_to_percent_changes': {
        'required': [],
        'optional': ['decimal_places'],
        'types': {
            'decimal_places': int
        },
        'validators': {
            'decimal_places': lambda x: isinstance(x, int) and x >= 0
        }
    },
    'add_rand_to_data_points': {
        'required': ['rand_size'],
        'optional': [],
        'types': {
            'rand_size': int
        },
        'validators': {
            'rand_size': lambda x: isinstance(x, int) and 1 <= x <= 3
        }
    }
}


def validate_function_arguments(function_name: str, args: Dict[str, Any]) -> bool:
    """Validate arguments for built-in processing functions.

    Args:
        function_name: Name of the function to validate.
        args: Dictionary of arguments to validate.

    Returns:
        True if valid.

    Raises:
        ValueError: If validation fails.
        TypeError: If argument types are incorrect.
    """
    if function_name not in BUILTIN_FUNCTION_VALIDATION:
        return True  # External functions are not validated

    schema = BUILTIN_FUNCTION_VALIDATION[function_name]

    for req_arg in schema['required']:
        if req_arg not in args:
            raise ValueError(f"Missing required argument '{req_arg}' for function '{function_name}'")

    allowed_args = set(schema['required'] + schema['optional'])
    provided_args = set(args.keys())
    unknown_args = provided_args - allowed_args
    if unknown_args:
        raise ValueError(f"Unknown arguments for function '{function_name}': {unknown_args}")

    for arg_name, arg_value in args.items():
        if arg_name in schema['types']:
            expected_type = schema['types'][arg_name]
            if not isinstance(arg_value, expected_type):
                type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                raise TypeError(f"Argument '{arg_name}' for function '{function_name}' must be {type_name}, got {type(arg_value).__name__}")

        if arg_name in schema['validators']:
            validator = schema['validators'][arg_name]
            if not validator(arg_value):
                raise ValueError(f"Invalid value for argument '{arg_name}' in function '{function_name}': {arg_value}")

    return True


def get_function_info(function_name: str) -> Dict[str, Any]:
    """Get information about a function (built-in or external).

    Args:
        function_name: Function name to inspect.

    Returns:
        Dictionary with function information.
    """
    try:
        func = resolve_function(function_name)
        return {
            'name': function_name,
            'type': 'builtin' if function_name in builtin_processing_functions else 'external',
            'callable': callable(func),
            'module': getattr(func, '__module__', 'unknown'),
            'doc': getattr(func, '__doc__', 'No documentation available'),
            'exists': True
        }
    except Exception as e:
        return {
            'name': function_name,
            'type': 'unknown',
            'callable': False,
            'module': 'unknown',
            'doc': 'Function not found',
            'exists': False,
            'error': str(e)
        }