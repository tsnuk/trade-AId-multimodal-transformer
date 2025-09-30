"""
Analyze directional accuracy for Day of Week pattern with 39 repetitions per day.
This pattern should be trivially easy for a transformer to learn.
"""

def analyze_repeated_day_pattern():
    """Analyze the actual Day of Week pattern with 39 repetitions."""
    print("=== ANALYZING REPEATED DAY OF WEEK PATTERN ===\n")

    # Actual pattern: each day repeats 39 times
    # [1,1,1,...,1, 2,2,2,...,2, 3,3,3,...,3, 4,4,4,...,4, 5,5,5,...,5, 1,1,1,...,1]
    #   39 times    39 times    39 times    39 times    39 times    39 times...

    repetitions_per_day = 39
    vocab = [1, 2, 3, 4, 5]

    # Generate the actual sequence pattern
    sequence = []
    for day in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]:  # Two full cycles for example
        sequence.extend([day] * repetitions_per_day)

    print(f"Example sequence pattern (first 50 elements):")
    print(f"  {sequence[:50]}...")
    print(f"Total length: {len(sequence)} elements")

    # Analyze transitions and directional expectations
    print(f"\n=== TRANSITION ANALYSIS ===")

    transitions = []
    for i in range(len(sequence) - 1):
        prev_val = sequence[i]
        next_val = sequence[i + 1]

        # Calculate direction
        if next_val > prev_val:
            direction = 1  # upward
        elif next_val < prev_val:
            direction = -1  # downward
        else:
            direction = 0  # flat

        transitions.append((prev_val, next_val, direction))

    # Count transition types
    flat_transitions = sum(1 for _, _, d in transitions if d == 0)
    upward_transitions = sum(1 for _, _, d in transitions if d == 1)
    downward_transitions = sum(1 for _, _, d in transitions if d == -1)
    total_transitions = len(transitions)

    print(f"Transition statistics:")
    print(f"  Flat (same day): {flat_transitions}/{total_transitions} ({flat_transitions/total_transitions*100:.1f}%)")
    print(f"  Upward (day change): {upward_transitions}/{total_transitions} ({upward_transitions/total_transitions*100:.1f}%)")
    print(f"  Downward (5->1): {downward_transitions}/{total_transitions} ({downward_transitions/total_transitions*100:.1f}%)")

    # Show specific transition examples
    print(f"\nSpecific transition examples:")
    day_change_transitions = [(p, n, d) for p, n, d in transitions if d != 0][:10]
    for prev, next_val, direction in day_change_transitions:
        dir_name = "UP" if direction == 1 else ("DOWN" if direction == -1 else "FLAT")
        print(f"  {prev} -> {next_val} ({dir_name})")

    print(f"\n=== PERFECT MODEL EXPECTATIONS ===")

    # For a perfect model learning this pattern:
    # - Most predictions should be flat (same day continues)
    # - Occasional upward transitions (day changes from N to N+1)
    # - Rare downward transitions (Friday to Monday: 5->1)

    perfect_flat_accuracy = flat_transitions / total_transitions * 100
    perfect_upward_accuracy = upward_transitions / total_transitions * 100
    perfect_downward_accuracy = downward_transitions / total_transitions * 100

    print(f"A perfect model should achieve:")
    print(f"  Overall directional accuracy: ~100% (pattern is deterministic)")
    print(f"  Breakdown:")
    print(f"    - Flat predictions: {perfect_flat_accuracy:.1f}% of the time")
    print(f"    - Upward predictions: {perfect_upward_accuracy:.1f}% of the time")
    print(f"    - Downward predictions: {perfect_downward_accuracy:.1f}% of the time")

    return flat_transitions, upward_transitions, downward_transitions, total_transitions

def simulate_model_errors_on_repeated_pattern():
    """Simulate what happens when model makes errors on this pattern."""
    print(f"\n=== SIMULATING MODEL ERRORS ===")

    repetitions_per_day = 39

    # Common error scenarios for this pattern
    scenarios = [
        {
            "name": "Perfect learning",
            "description": "Model perfectly predicts the pattern",
            "error_rate": 0.0
        },
        {
            "name": "Occasional wrong day",
            "description": "Model sometimes predicts wrong day value",
            "error_rate": 0.1  # 10% errors
        },
        {
            "name": "Frequent day confusion",
            "description": "Model often predicts wrong day value",
            "error_rate": 0.3  # 30% errors
        },
        {
            "name": "Almost random",
            "description": "Model predictions are mostly random",
            "error_rate": 0.8  # 80% errors
        }
    ]

    # True pattern statistics (from previous analysis)
    # In a typical sequence: ~97.5% flat, ~2% upward, ~0.5% downward
    true_flat_pct = 97.5
    true_upward_pct = 2.0
    true_downward_pct = 0.5

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']} ({scenario['description']})")
        error_rate = scenario['error_rate']
        correct_rate = 1.0 - error_rate

        # Calculate expected directional accuracy
        # When model is correct: 100% directional accuracy for that prediction
        # When model is wrong: depends on what it predicts

        # Simplified calculation: assume errors are somewhat random
        # For flat transitions (97.5% of data):
        #   - Correct: direction = 0, accuracy = 100%
        #   - Wrong: random direction, accuracy ~ 20% (since true is 0, random has 20% chance of being 0)

        # For upward transitions (2% of data):
        #   - Correct: direction = 1, accuracy = 100%
        #   - Wrong: random direction, accuracy ~ 40% (random has 40% chance of being 1)

        # For downward transitions (0.5% of data):
        #   - Correct: direction = -1, accuracy = 100%
        #   - Wrong: random direction, accuracy ~ 40% (random has 40% chance of being -1)

        expected_directional_accuracy = (
            (true_flat_pct/100) * (correct_rate * 1.0 + error_rate * 0.2) +
            (true_upward_pct/100) * (correct_rate * 1.0 + error_rate * 0.4) +
            (true_downward_pct/100) * (correct_rate * 1.0 + error_rate * 0.4)
        ) * 100

        print(f"  Value error rate: {error_rate*100:.0f}%")
        print(f"  Expected directional accuracy: {expected_directional_accuracy:.1f}%")

def main():
    flat, upward, downward, total = analyze_repeated_day_pattern()
    simulate_model_errors_on_repeated_pattern()

    print(f"\n=== CONCLUSION ===")
    print(f"Your 40% validation directional accuracy suggests the model is making")
    print(f"MASSIVE errors on what should be a trivially easy pattern!")
    print(f"")
    print(f"Possible issues to investigate:")
    print(f"1. Data preprocessing: Is the Day of Week data actually this pattern?")
    print(f"2. Model capacity: Is the model too small/simple?")
    print(f"3. Training: Has the model been trained enough?")
    print(f"4. Evaluation: Are we evaluating on the right data?")
    print(f"5. Bug: Is there a bug in data loading or model training?")

if __name__ == "__main__":
    main()