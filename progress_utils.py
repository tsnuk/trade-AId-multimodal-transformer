"""
Progress indicator utilities for the multimodal transformer system.

Provides simple, reliable progress bars that work across different terminal environments.
"""

import sys
import time


def show_progress_bar(current, total, message="Progress", bar_width=40):
    """
    Display a progress bar with percentage and message.

    Args:
        current (int): Current progress value (0 to total)
        total (int): Total/maximum value
        message (str): Message to display before the progress bar
        bar_width (int): Width of the progress bar in characters
    """
    if total == 0:
        percent = 100.0
    else:
        percent = (current / total) * 100

    filled = int((current / total) * bar_width) if total > 0 else bar_width
    bar = '#' * filled + '-' * (bar_width - filled)

    sys.stdout.write(f'\r{message}: [{bar}] {percent:5.1f}% ({current}/{total})')
    sys.stdout.flush()

    if current >= total:
        print()  # New line when complete


def show_spinner(message="Processing", step=None):
    """
    Show a simple spinning cursor for unknown duration tasks.

    Args:
        message (str): Message to display
        step (int): Current step number for cycling the spinner
    """
    if step is None:
        step = int(time.time() * 4)  # Auto-cycle based on time

    spinner_chars = ['|', '/', '-', '\\']
    spinner = spinner_chars[step % 4]

    sys.stdout.write(f'\r{message} {spinner}')
    sys.stdout.flush()


def show_stage_progress(stage_num, total_stages, stage_name, substep=None, total_substeps=None):
    """
    Show multi-stage progress with optional substep progress.

    Args:
        stage_num (int): Current stage number (1-based)
        total_stages (int): Total number of stages
        stage_name (str): Name of current stage
        substep (int, optional): Current substep within stage
        total_substeps (int, optional): Total substeps in current stage
    """
    stage_info = f'[{stage_num}/{total_stages}] {stage_name}'

    if substep is not None and total_substeps is not None:
        substep_percent = (substep / total_substeps) * 100 if total_substeps > 0 else 100
        sys.stdout.write(f'\r{stage_info}: {substep_percent:5.1f}% ({substep}/{total_substeps})')
    else:
        sys.stdout.write(f'\r{stage_info}...')

    sys.stdout.flush()


def finish_progress_line():
    """Print a newline to finish the current progress line."""
    print()


def clear_progress_line():
    """Clear the current progress line."""
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()


def show_completion_animation(message="Complete", duration=1.0):
    """
    Show a brief animated completion indicator.

    Args:
        message (str): Message to display
        duration (float): Animation duration in seconds
    """
    frames = ['●', '◐', '◑', '◒', '◓', '●', '◐', '◑', '◒', '◓', '✓']
    frame_delay = duration / len(frames)

    for frame in frames:
        sys.stdout.write(f'\r{message} {frame}')
        sys.stdout.flush()
        time.sleep(frame_delay)

    print()  # New line when complete


# Example usage functions for testing
def demo_progress_bar():
    """Demo the progress bar functionality."""
    print("Demo: Progress Bar")
    total = 50
    for i in range(total + 1):
        show_progress_bar(i, total, "Loading data")
        time.sleep(0.05)
    print("Complete!")


def demo_spinner():
    """Demo the spinner functionality."""
    print("Demo: Spinner")
    for i in range(20):
        show_spinner("Processing", i)
        time.sleep(0.2)
    print("\rProcessing complete!        ")


def demo_stage_progress():
    """Demo the multi-stage progress functionality."""
    print("Demo: Multi-stage Progress")
    stages = ["Loading files", "Processing data", "Building vocabularies", "Initializing model"]

    for stage_idx, stage_name in enumerate(stages, 1):
        substeps = 10
        for substep in range(substeps + 1):
            show_stage_progress(stage_idx, len(stages), stage_name, substep, substeps)
            time.sleep(0.1)
        finish_progress_line()

    print("All stages complete!")


if __name__ == "__main__":
    # Run demos if script is executed directly
    demo_progress_bar()
    print()
    demo_spinner()
    print()
    demo_stage_progress()