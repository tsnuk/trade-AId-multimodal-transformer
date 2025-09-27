"""Configuration utility functions for the multimodal transformer system.

Provides centralized access to system configuration with caching to avoid
repeated imports and improve performance.
"""

# Global configuration cache - will be populated when first accessed
_config_cache = None


def _get_config():
    """Lazy load system configuration through compatibility layer.

    Caches configuration on first access to avoid repeated imports.

    Returns:
        dict: System configuration dictionary containing device, batch_size,
              block_size, eval_iters, and other training parameters.
    """
    global _config_cache
    if _config_cache is None:
        from compatibility_layer import get_system_configuration
        _config_cache = get_system_configuration()
    return _config_cache


# Create property-like accessors for configuration values
def _get_device():
    """Get the configured device (cpu/cuda/auto)."""
    return _get_config()['device']


def _get_block_size():
    """Get the configured block size for sequence length."""
    return _get_config()['block_size']


def _get_batch_size():
    """Get the configured batch size for training."""
    return _get_config()['batch_size']


def _get_eval_iters():
    """Get the configured number of evaluation iterations."""
    return _get_config()['eval_iters']


def _get_n_embd():
    """Get the configured embedding dimension."""
    return _get_config()['n_embd']


def _get_n_head():
    """Get the configured number of attention heads."""
    return _get_config()['n_head']


def _get_n_layer():
    """Get the configured number of transformer layers."""
    return _get_config()['n_layer']


def _get_dropout():
    """Get the configured dropout rate."""
    return _get_config()['dropout']


def _get_fixed_values():
    """Get the configured fixed values for discretization."""
    return _get_config()['fixed_values']