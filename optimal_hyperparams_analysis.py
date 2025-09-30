"""
Calculate optimal hyperparameters for learning Day of Week pattern
where each day repeats 39 times before transitioning.
"""

def analyze_block_size_requirements():
    """Analyze what block_size is needed to learn the Day of Week pattern."""
    print("=== BLOCK SIZE ANALYSIS FOR DAY OF WEEK LEARNING ===\n")

    repetitions_per_day = 39
    num_days = 5

    print(f"Pattern: Each day repeats {repetitions_per_day} times")
    print(f"Days in cycle: {num_days}")
    print(f"Full cycle length: {repetitions_per_day * num_days} = {repetitions_per_day * num_days} elements")

    # Current situation
    current_block_size = 64
    print(f"\nCURRENT SITUATION (block_size = {current_block_size}):")

    # How many days fit in a block?
    days_in_block = current_block_size / repetitions_per_day
    print(f"  Days that fit in block: {days_in_block:.2f}")
    print(f"  Most blocks contain: {int(days_in_block)} full day(s) + partial day")

    # Probability of seeing a transition
    # Transition happens every 39 elements
    # In a 64-element block, probability of capturing a transition:
    transition_probability = min(1.0, (current_block_size - repetitions_per_day + 1) / repetitions_per_day)
    print(f"  Probability of seeing day transition in block: {transition_probability:.2f}")

    # Recommended block sizes
    print(f"\nRECOMMENDED BLOCK SIZES:")

    recommendations = [
        {
            "size": repetitions_per_day + 10,
            "description": "Minimum to see one transition",
            "reasoning": "Ensures most blocks contain at least one day change"
        },
        {
            "size": repetitions_per_day * 2,
            "description": "Optimal for learning transitions",
            "reasoning": "Contains 2 full days, always shows transition pattern"
        },
        {
            "size": repetitions_per_day * 3,
            "description": "Rich context",
            "reasoning": "Contains 3 days, shows multiple transitions"
        },
        {
            "size": repetitions_per_day * 5,
            "description": "Full cycle context",
            "reasoning": "Contains complete 5-day cycle"
        }
    ]

    for rec in recommendations:
        size = rec["size"]
        days_in_block = size / repetitions_per_day
        transitions_in_block = max(0, int(days_in_block) - 1)

        print(f"\n  Block size {size}:")
        print(f"    {rec['description']}")
        print(f"    Contains: {days_in_block:.1f} days")
        print(f"    Guaranteed transitions per block: {transitions_in_block}")
        print(f"    Reasoning: {rec['reasoning']}")

def calculate_training_requirements():
    """Calculate how much training is needed."""
    print(f"\n=== TRAINING REQUIREMENTS ANALYSIS ===\n")

    repetitions_per_day = 39

    # For a simple pattern like this, the model should learn very quickly
    print("Expected learning timeline:")
    print("- Simple repetition pattern (same number 39 times)")
    print("- Should converge within 100-500 iterations")
    print("- Pattern is deterministic and very regular")

    # Calculate effective training examples needed
    transitions_per_cycle = 5  # 1->2, 2->3, 3->4, 4->5, 5->1

    print(f"\nTraining data requirements:")
    print(f"- Need to see each of {transitions_per_cycle} transition types")
    print(f"- With proper block_size, should learn within first few epochs")
    print(f"- Current pattern is much simpler than typical NLP tasks")

def recommend_hyperparameters():
    """Provide specific hyperparameter recommendations."""
    print(f"\n=== HYPERPARAMETER RECOMMENDATIONS ===\n")

    print("CRITICAL CHANGE - Block Size:")
    print("  Current: block_size = 64")
    print("  Recommended: block_size = 78 (39*2)")
    print("  Alternative: block_size = 117 (39*3)")
    print("  Reasoning: Must span at least 2 full days to see transitions")

    print(f"\nOther hyperparameters:")
    print("  batch_size: Keep current (32) - should be fine")
    print("  learning_rate: 0.001-0.003 (standard range)")
    print("  max_iters: 1000-2000 (should be plenty for this simple pattern)")
    print("  eval_interval: 100 (check progress frequently)")

    print(f"\nModel architecture:")
    print("  n_embd: 64-128 (simple pattern doesn't need large embeddings)")
    print("  n_head: 4-8 (modest attention)")
    print("  n_layer: 2-4 (shallow network should suffice)")
    print("  dropout: 0.1-0.2 (prevent overfitting)")

    print(f"\nEXPECTED RESULTS with proper block_size:")
    print("  - Training loss should drop quickly (within 100-200 iterations)")
    print("  - Directional accuracy should reach 95%+ within 500 iterations")
    print("  - Pattern is so simple that perfect learning is expected")

def analyze_current_vs_optimal():
    """Compare current setup vs optimal setup."""
    print(f"\n=== CURRENT VS OPTIMAL COMPARISON ===\n")

    print("CURRENT SETUP PROBLEMS:")
    print("  ERROR: block_size=64 < day_length=39*2")
    print("  ERROR: Most sequences see only 1-2 days (no transitions)")
    print("  ERROR: Model can't learn transition pattern")
    print("  ERROR: Results in random-like performance (40%)")

    print(f"\nOPTIMAL SETUP BENEFITS:")
    print("  OK: block_size=78+ spans multiple days")
    print("  OK: Every sequence shows day transitions")
    print("  OK: Model can learn the repetition + transition pattern")
    print("  OK: Should achieve 95%+ directional accuracy")

    print(f"\nQUICK TEST CONFIGURATION:")
    print("  block_size: 78")
    print("  batch_size: 16 (smaller for memory)")
    print("  max_iters: 500")
    print("  eval_interval: 50")
    print("  learning_rate: 0.002")

if __name__ == "__main__":
    analyze_block_size_requirements()
    calculate_training_requirements()
    recommend_hyperparameters()
    analyze_current_vs_optimal()