#!/usr/bin/env python3
"""
Quick analysis: What % of responses have all operations with explicit token positions?
"""
import json
from pathlib import Path
from collections import Counter

def is_operation_probeable(op: dict) -> bool:
    """Check if operation has all token positions found."""
    positions = [
        op.get('operand1_positions', [-1])[0],
        op.get('operator_positions', [-1])[0],
        op.get('operand2_positions', [-1])[0],
        op.get('result_positions', [-1])[0],
    ]
    return all(p >= 0 for p in positions)

def analyze_response(resp: dict) -> dict:
    """Analyze a single response for probeability."""
    ops = resp.get('operations', [])
    if not ops:
        return {
            'has_ops': False,
            'num_ops': 0,
            'all_probeable': False,
            'probeable_ops': 0,
            'final_correct': resp.get('final_correct'),
        }

    probeable_ops = sum(1 for op in ops if is_operation_probeable(op))

    return {
        'has_ops': True,
        'num_ops': len(ops),
        'all_probeable': probeable_ops == len(ops),
        'probeable_ops': probeable_ops,
        'final_correct': resp.get('final_correct'),
    }

def main():
    # Load analyzed responses
    data_path = Path(__file__).parent / "responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json"

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Analyze all responses
    stats = {
        'total_examples': len(data),
        'total_responses': 0,
        'responses_with_ops': 0,
        'responses_all_probeable': 0,
        'total_operations': 0,
        'probeable_operations': 0,
        'examples_with_probeable_response': 0,
        'correct_and_probeable': 0,
    }

    ops_per_response = Counter()
    probeable_per_example = Counter()

    for example in data:
        responses = example.get('responses', [])
        example_has_probeable = False

        for resp in responses:
            stats['total_responses'] += 1
            analysis = analyze_response(resp)

            if analysis['has_ops']:
                stats['responses_with_ops'] += 1
                stats['total_operations'] += analysis['num_ops']
                stats['probeable_operations'] += analysis['probeable_ops']
                ops_per_response[analysis['num_ops']] += 1

                if analysis['all_probeable']:
                    stats['responses_all_probeable'] += 1
                    example_has_probeable = True

                    if analysis['final_correct']:
                        stats['correct_and_probeable'] += 1

        if example_has_probeable:
            stats['examples_with_probeable_response'] += 1

        # Count how many probeable responses per example
        probeable_count = sum(
            1 for resp in responses
            if analyze_response(resp)['all_probeable']
        )
        probeable_per_example[probeable_count] += 1

    # Print results
    print("\n" + "="*60)
    print("PROBEABILITY ANALYSIS")
    print("="*60)

    print(f"\n📊 Overall Stats:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Responses per example: {stats['total_responses'] / stats['total_examples']:.1f}")

    print(f"\n📈 Operations Found:")
    print(f"   Responses with any ops: {stats['responses_with_ops']} ({100*stats['responses_with_ops']/stats['total_responses']:.1f}%)")
    print(f"   Total operations found: {stats['total_operations']}")
    print(f"   Avg ops per response (when found): {stats['total_operations']/max(stats['responses_with_ops'],1):.2f}")

    print(f"\n✅ Probeability:")
    print(f"   Responses ALL ops probeable: {stats['responses_all_probeable']} ({100*stats['responses_all_probeable']/stats['total_responses']:.1f}%)")
    print(f"   Probeable operations: {stats['probeable_operations']} ({100*stats['probeable_operations']/max(stats['total_operations'],1):.1f}%)")
    print(f"   Examples with ≥1 probeable response: {stats['examples_with_probeable_response']} ({100*stats['examples_with_probeable_response']/stats['total_examples']:.1f}%)")

    print(f"\n🎯 Correct AND Probeable:")
    print(f"   Responses: {stats['correct_and_probeable']} ({100*stats['correct_and_probeable']/stats['total_responses']:.1f}%)")

    print(f"\n📊 Operations per response distribution:")
    for n_ops, count in sorted(ops_per_response.items()):
        print(f"   {n_ops} ops: {count} responses")

    print(f"\n📊 Probeable responses per example:")
    for n_prob, count in sorted(probeable_per_example.items()):
        pct = 100 * count / stats['total_examples']
        print(f"   {n_prob} probeable: {count} examples ({pct:.1f}%)")

    # Estimate dataset size for probing
    print(f"\n🔮 Estimated Probing Dataset Size:")
    avg_ops = stats['total_operations'] / max(stats['responses_with_ops'], 1)
    estimated_probe_samples = stats['responses_all_probeable'] * avg_ops
    print(f"   Probeable responses: {stats['responses_all_probeable']}")
    print(f"   Avg ops per response: {avg_ops:.2f}")
    print(f"   Estimated probe samples: ~{int(estimated_probe_samples)}")

if __name__ == '__main__':
    main()
