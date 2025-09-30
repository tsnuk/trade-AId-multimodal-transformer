"""
Comprehensive debugging script for directional accuracy analysis.
This script will trace through the exact calculations to identify why
Day of Week directional accuracy is unexpectedly low.
"""

import torch
import torch.nn.functional as F
import numbers

def debug_get_direction_sign(current_value, previous_value, is_percentage_data, debug_info=""):
    """Debug version of _get_direction_sign with detailed logging."""
    print(f"  {debug_info}")
    print(f"    current_value: {current_value}, previous_value: {previous_value}, is_percentage: {is_percentage_data}")

    if is_percentage_data:
        if current_value > 0:
            direction = 1
        elif current_value < 0:
            direction = -1
        else:
            direction = 0
        print(f"    -> Direction (percentage): {direction}")
        return direction
    else:
        # For value data, direction is based on change from previous value
        if not isinstance(previous_value, numbers.Number):
            print(f"    -> Direction: None (previous_value not numeric)")
            return None

        change = current_value - previous_value
        if change > 0:
            direction = 1
        elif change < 0:
            direction = -1
        else:
            direction = 0
        print(f"    -> Change: {change}, Direction: {direction}")
        return direction

def simulate_day_of_week_directional_calculation():
    """Simulate directional calculations for Day of Week pattern."""
    print("=== SIMULATING DAY OF WEEK DIRECTIONAL CALCULATIONS ===\n")

    # Day of Week vocabulary: [1, 2, 3, 4, 5] (assuming 5-day work week)
    vocab = [1, 2, 3, 4, 5]
    is_percentage_data = False

    # Common transitions in Day of Week data
    test_cases = [
        {"previous": 1, "actual": 2, "predicted": 2, "description": "Mon->Tue, predict Tue"},
        {"previous": 2, "actual": 3, "predicted": 3, "description": "Tue->Wed, predict Wed"},
        {"previous": 3, "actual": 4, "predicted": 4, "description": "Wed->Thu, predict Thu"},
        {"previous": 4, "actual": 5, "predicted": 5, "description": "Thu->Fri, predict Fri"},
        {"previous": 5, "actual": 1, "predicted": 1, "description": "Fri->Mon, predict Mon (CRITICAL CASE)"},
        {"previous": 5, "actual": 1, "predicted": 2, "description": "Fri->Mon, but predict Tue (WRONG VALUE)"},
        {"previous": 5, "actual": 1, "predicted": 5, "description": "Fri->Mon, but predict Fri (REPEAT)"},
        {"previous": 4, "actual": 5, "predicted": 1, "description": "Thu->Fri, but predict Mon (WRONG CYCLE)"},
    ]

    wins = 0
    losses = 0

    for i, case in enumerate(test_cases):
        print(f"Test Case {i+1}: {case['description']}")

        # Calculate actual direction
        actual_direction = debug_get_direction_sign(
            case['actual'], case['previous'], is_percentage_data,
            "ACTUAL direction"
        )

        # Calculate predicted direction
        predicted_direction = debug_get_direction_sign(
            case['predicted'], case['previous'], is_percentage_data,
            "PREDICTED direction"
        )

        # Compare
        if actual_direction is not None and predicted_direction is not None:
            if actual_direction == predicted_direction:
                result = "WIN"
                wins += 1
            else:
                result = "LOSS"
                losses += 1
        else:
            result = "SKIP (None direction)"

        print(f"    RESULT: {result}")
        print(f"    Match: actual_dir={actual_direction}, predicted_dir={predicted_direction}")
        print()

    total = wins + losses
    accuracy = (wins / total * 100) if total > 0 else 0
    print(f"SIMULATION RESULTS:")
    print(f"  Wins: {wins}, Losses: {losses}, Total: {total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print()

def analyze_vocabulary_directional_distribution(vocab, is_percentage_data=False):
    """Analyze what directional predictions are possible given a vocabulary."""
    print("=== VOCABULARY DIRECTIONAL ANALYSIS ===\n")
    print(f"Vocabulary: {vocab}")
    print(f"Is percentage data: {is_percentage_data}")

    # Test all possible previous->current transitions
    direction_counts = {-1: 0, 0: 0, 1: 0}
    transitions = []

    for prev_val in vocab:
        for curr_val in vocab:
            if is_percentage_data:
                if curr_val > 0:
                    direction = 1
                elif curr_val < 0:
                    direction = -1
                else:
                    direction = 0
            else:
                change = curr_val - prev_val
                if change > 0:
                    direction = 1
                elif change < 0:
                    direction = -1
                else:
                    direction = 0

            direction_counts[direction] += 1
            transitions.append(f"{prev_val}->{curr_val} (dir: {direction})")

    print("All possible transitions:")
    for transition in transitions[:20]:  # Show first 20
        print(f"  {transition}")
    if len(transitions) > 20:
        print(f"  ... and {len(transitions) - 20} more")

    print(f"\nDirection distribution:")
    total_transitions = sum(direction_counts.values())
    for direction, count in direction_counts.items():
        pct = (count / total_transitions * 100) if total_transitions > 0 else 0
        dir_name = {-1: "Downward", 0: "Flat", 1: "Upward"}[direction]
        print(f"  {dir_name} ({direction}): {count}/{total_transitions} ({pct:.1f}%)")

    print()

def debug_model_prediction_example():
    """Debug a specific model prediction scenario."""
    print("=== DEBUGGING MODEL PREDICTION SCENARIO ===\n")

    # Simulate model logits for Day of Week vocabulary [1,2,3,4,5]
    vocab = [1, 2, 3, 4, 5]

    # Example: Previous value was 5 (Friday), actual next should be 1 (Monday)
    previous_value = 5
    actual_value = 1

    # Simulate different model prediction scenarios
    scenarios = [
        {"logits": [3.0, 0.5, 0.1, 0.1, 0.1], "description": "Model strongly predicts 1 (correct value)"},
        {"logits": [0.1, 3.0, 0.1, 0.1, 0.1], "description": "Model strongly predicts 2 (wrong value, but same direction)"},
        {"logits": [0.1, 0.1, 0.1, 0.1, 3.0], "description": "Model strongly predicts 5 (repeat previous, wrong direction)"},
        {"logits": [1.0, 1.0, 1.0, 1.0, 1.0], "description": "Model is uncertain (uniform distribution)"},
    ]

    print(f"Scenario: Previous = {previous_value}, Actual next = {actual_value}")
    actual_direction = debug_get_direction_sign(actual_value, previous_value, False, "Actual direction")

    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['description']}")
        logits = torch.tensor(scenario['logits'])
        probs = F.softmax(logits, dim=0)
        predicted_idx = torch.argmax(logits).item()
        predicted_value = vocab[predicted_idx]

        print(f"  Predicted value: {predicted_value} (index {predicted_idx})")
        print(f"  Probability distribution: {[f'{p:.3f}' for p in probs.tolist()]}")

        predicted_direction = debug_get_direction_sign(predicted_value, previous_value, False, "Predicted direction")

        if actual_direction == predicted_direction:
            result = "DIRECTIONAL WIN"
        else:
            result = "DIRECTIONAL LOSS"

        print(f"  Result: {result}")
        print(f"  Actual dir: {actual_direction}, Predicted dir: {predicted_direction}")

if __name__ == "__main__":
    # Run all debugging tests
    simulate_day_of_week_directional_calculation()
    analyze_vocabulary_directional_distribution([1, 2, 3, 4, 5], False)
    analyze_vocabulary_directional_distribution([-2.5, -1.0, 0.0, 1.0, 2.5], True)  # Example percentage vocab
    debug_model_prediction_example()