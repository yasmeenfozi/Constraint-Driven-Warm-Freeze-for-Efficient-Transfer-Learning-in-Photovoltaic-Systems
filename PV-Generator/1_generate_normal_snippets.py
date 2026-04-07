"""Generate 10-second normal voltage snippets."""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sims.pv_simulator_highrate import build_highrate_dataset
import config


SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
NORMAL_DIR = os.path.join(DATA_DIR, "normal")
os.makedirs(NORMAL_DIR, exist_ok=True)

SNIPPET_DURATION_S = 10
TARGET_HZ = 30
TARGET_SAMPLES_PER_SNIPPET = SNIPPET_DURATION_S * TARGET_HZ
N_SNIPPETS = 14400
SEED = 42


def main():
    print("=" * 60)
    print("GENERATING NORMAL 10-SECOND SNIPPETS")
    print("=" * 60)

    print(f"\nGenerating {N_SNIPPETS:,} snippets")
    print(f"  Duration: {SNIPPET_DURATION_S} s")
    print(f"  Target sampling rate: {TARGET_HZ} Hz")
    print(f"  Target samples per snippet: {TARGET_SAMPLES_PER_SNIPPET}")
    print(f"  Total duration: {N_SNIPPETS * SNIPPET_DURATION_S / 3600:.1f} hours")

    total_seconds = N_SNIPPETS * SNIPPET_DURATION_S
    print(f"\nGenerating {total_seconds / 3600:.1f} hours of continuous data...")

    config.SEED = SEED
    df = build_highrate_dataset(out_seconds=total_seconds)

    voltage = df["pv_v"].values
    actual_hz = len(voltage) / total_seconds

    print(f"  Generated samples: {len(voltage):,}")
    print(f"  Actual sampling rate: {actual_hz:.1f} Hz")

    samples_per_snippet = TARGET_SAMPLES_PER_SNIPPET
    if actual_hz != TARGET_HZ:
        print(f"\nWarning: expected {TARGET_HZ} Hz but got {actual_hz:.1f} Hz")
        samples_per_snippet = int(SNIPPET_DURATION_S * actual_hz)
        print(f"Adjusted samples per snippet: {samples_per_snippet}")

    print(f"\nSplitting into snippets and saving to {NORMAL_DIR}")

    snippet_count = 0
    for start in range(0, len(voltage) - samples_per_snippet + 1, samples_per_snippet):
        snippet = voltage[start:start + samples_per_snippet]

        if len(snippet) != samples_per_snippet:
            continue

        output_path = os.path.join(NORMAL_DIR, f"n_{snippet_count}.npy")
        np.save(output_path, snippet)
        snippet_count += 1

        if snippet_count % 500 == 0:
            print(f"  Saved {snippet_count:,} snippets")

        if snippet_count >= N_SNIPPETS:
            break

    print(f"\nGenerated {snippet_count:,} normal snippets")
    print(f"Saved to: {NORMAL_DIR}")
    print(f"Snippet shape: ({samples_per_snippet},) per file")
    print("=" * 60)


if __name__ == "__main__":
    main()