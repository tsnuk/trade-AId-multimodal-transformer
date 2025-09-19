"""processing_pipeline.py

Processing pipeline engine for executing sequential data transformations.

This module provides the core pipeline execution engine that processes data through
a sequence of functions, where each function receives the output of the previous
function. Supports both built-in and external processing functions.

Features:
- Sequential data flow through processing steps
- Built-in and external function resolution
- Error handling and validation
- Execution tracking and metadata
- Support for function arguments and configuration
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from dataclasses import dataclass, field
from processing_registry import resolve_function
from schema import ProcessingStep, InputSchema

# Set up logging for pipeline execution
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of processing pipeline execution"""
    processed_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)
    successful_steps: int = 0
    total_steps: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the pipeline executed successfully"""
        return self.error is None

    @property
    def completion_percentage(self) -> float:
        """Percentage of steps completed successfully"""
        if self.total_steps == 0:
            return 100.0
        return (self.successful_steps / self.total_steps) * 100.0


class ProcessingPipeline:
    """
    Processing pipeline engine that executes sequential data transformations.

    The pipeline processes data through a sequence of functions where:
    1. Each function receives the output of the previous function
    2. Functions can be built-in (simple name) or external (fully qualified name)
    3. Function arguments are passed via the 'args' parameter
    4. Execution is tracked and logged for debugging
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the processing pipeline.

        Args:
            enable_logging: Whether to enable detailed execution logging
        """
        self.enable_logging = enable_logging
        self.execution_history: List[PipelineResult] = []

    def execute(self, initial_data: Any, processing_steps: List[ProcessingStep],
                modality_name: str = "Unknown") -> PipelineResult:
        """
        Execute a processing pipeline on the given data.

        Args:
            initial_data: The initial data to process
            processing_steps: List of processing steps to execute sequentially
            modality_name: Name of the modality for logging purposes

        Returns:
            PipelineResult containing processed data and execution metadata
        """
        result = PipelineResult(
            processed_data=initial_data,
            total_steps=len([step for step in processing_steps if step.enabled])
        )

        if not processing_steps:
            result.execution_log.append("No processing steps defined - returning original data")
            return result

        # Filter enabled steps
        enabled_steps = [step for step in processing_steps if step.enabled]
        if not enabled_steps:
            result.execution_log.append("No enabled processing steps - returning original data")
            return result

        current_data = initial_data

        try:
            for i, step in enumerate(enabled_steps):
                step_name = f"Step {i+1}: {step.function}"

                if self.enable_logging:
                    logger.info(f"Executing {step_name} for modality '{modality_name}'")

                # Resolve the function
                try:
                    function = resolve_function(step.function)
                    result.execution_log.append(f"✓ {step_name} - Function resolved successfully")
                except Exception as e:
                    error_msg = f"✗ {step_name} - Failed to resolve function: {e}"
                    result.execution_log.append(error_msg)
                    result.error = error_msg
                    logger.error(error_msg)
                    break

                # Execute the function
                try:
                    # Apply the function with the current data and step arguments
                    current_data = function(current_data, **step.args)
                    result.successful_steps += 1

                    # Log successful execution
                    args_str = f" with args {step.args}" if step.args else ""
                    success_msg = f"✓ {step_name} - Executed successfully{args_str}"
                    result.execution_log.append(success_msg)

                    if self.enable_logging:
                        data_info = self._get_data_info(current_data)
                        logger.info(f"{step_name} completed. {data_info}")

                except Exception as e:
                    error_msg = f"✗ {step_name} - Execution failed: {e}"
                    result.execution_log.append(error_msg)
                    result.error = error_msg
                    logger.error(error_msg)
                    break

            # Set final processed data
            result.processed_data = current_data

            # Update metadata
            result.metadata.update({
                'modality_name': modality_name,
                'initial_data_type': type(initial_data).__name__,
                'final_data_type': type(current_data).__name__,
                'steps_executed': result.successful_steps,
                'steps_total': result.total_steps
            })

            # Track special processing for later use
            self._track_special_processing(enabled_steps, result.metadata)

        except Exception as e:
            error_msg = f"Pipeline execution failed with unexpected error: {e}"
            result.execution_log.append(error_msg)
            result.error = error_msg
            logger.error(error_msg)

        # Store execution history
        self.execution_history.append(result)

        return result

    def execute_for_schema(self, initial_data: Any, schema: InputSchema) -> PipelineResult:
        """
        Execute processing pipeline defined in an InputSchema.

        Args:
            initial_data: The initial data to process
            schema: InputSchema containing processing steps and configuration

        Returns:
            PipelineResult containing processed data and execution metadata
        """
        return self.execute(initial_data, schema.processing_steps, schema.modality_name)

    def _get_data_info(self, data: Any) -> str:
        """Get informational string about the data for logging"""
        try:
            if isinstance(data, (list, tuple)):
                return f"Data type: {type(data).__name__}, Length: {len(data)}"
            elif hasattr(data, 'shape'):  # numpy arrays, tensors
                return f"Data type: {type(data).__name__}, Shape: {data.shape}"
            else:
                return f"Data type: {type(data).__name__}"
        except:
            return f"Data type: {type(data).__name__}"

    def _track_special_processing(self, steps: List[ProcessingStep], metadata: Dict[str, Any]) -> None:
        """
        Track special processing functions that require different handling later.

        This maintains compatibility with existing code that checks for special cases
        like percentage data for directional success calculations.
        """
        # Track functions that need special handling
        special_functions = {
            'calculate_percent_changes': 'is_percent_data',
            'bin_numeric_data': 'is_binned_data',
            'range_numeric_data': 'is_ranged_data'
        }

        for step in steps:
            if step.function in special_functions and step.enabled:
                flag_name = special_functions[step.function]
                metadata[flag_name] = True

                # Store additional information for specific functions
                if step.function == 'bin_numeric_data':
                    metadata['num_bins'] = step.args.get('num_bins')
                elif step.function == 'range_numeric_data':
                    metadata['num_whole_digits'] = step.args.get('num_whole_digits')
                    metadata['decimal_places'] = step.args.get('decimal_places')

    def validate_pipeline(self, processing_steps: List[ProcessingStep]) -> Tuple[bool, List[str]]:
        """
        Validate that all functions in the pipeline can be resolved.

        Args:
            processing_steps: List of processing steps to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        enabled_steps = [step for step in processing_steps if step.enabled]

        for i, step in enumerate(enabled_steps):
            try:
                resolve_function(step.function)
            except Exception as e:
                errors.append(f"Step {i+1} ({step.function}): {e}")

        return len(errors) == 0, errors

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline executions"""
        if not self.execution_history:
            return {"total_executions": 0}

        successful = sum(1 for result in self.execution_history if result.success)
        total = len(self.execution_history)

        return {
            "total_executions": total,
            "successful_executions": successful,
            "failure_rate": (total - successful) / total * 100 if total > 0 else 0,
            "average_steps_per_execution": sum(r.total_steps for r in self.execution_history) / total if total > 0 else 0,
            "most_recent_execution": self.execution_history[-1].success if self.execution_history else None
        }

    def clear_history(self) -> None:
        """Clear execution history"""
        self.execution_history.clear()


# Global pipeline instance for use throughout the application
default_pipeline = ProcessingPipeline()


def execute_processing_pipeline(data: Any, schema: InputSchema) -> PipelineResult:
    """
    Convenience function to execute processing pipeline for a schema.

    Args:
        data: Initial data to process
        schema: InputSchema containing processing configuration

    Returns:
        PipelineResult with processed data and metadata
    """
    return default_pipeline.execute_for_schema(data, schema)


def validate_schema_pipeline(schema: InputSchema) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a schema's processing pipeline.

    Args:
        schema: InputSchema to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    return default_pipeline.validate_pipeline(schema.processing_steps)