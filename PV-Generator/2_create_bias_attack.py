"""Create bias-attacked versions of the normal snippets."""

import os
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
NORMAL_DIR = os.path.join(DATA_DIR, "normal")
ATTACK_DIR = os.path.join(DATA_DIR, "attack")
os.makedirs(ATTACK_DIR, exist_ok=True)


def inject_bias_attack(
    window: np.ndarray,
    seed: int | None = None,
    min_bias: float = 0.003,
    max_bias: float = 0.008,
    noise_sigma: float = 0.002,
) -> np.ndarray:
    """
    Apply a bias attack over a random interior interval of the signal.

    The attack:
    - selects a random interval inside the window
    - applies a per-sample multiplicative bias with random sign
    - adds small local noise
    - applies smooth global background noise across the full window
    """
    rng = np.random.default_rng(seed)
    length = len(window)

    if length < 30:
        return window.copy()

    attacked = window.astype(np.float64).copy()

    fs = max(1, int(round(length / 10.0)))
    interior_start = fs
    interior_end = length - fs

    if interior_end <= interior_start + 1:
        return attacked

    start_idx = int(rng.integers(interior_start, interior_end - 1))
    end_idx = int(rng.integers(start_idx + 1, interior_end))

    min_interval_len = max(2, int(0.1 * (interior_end - interior_start)))
    if (end_idx - start_idx) < min_interval_len:
        end_idx = min(interior_end, start_idx + min_interval_len)

    chosen_indices = np.arange(start_idx, end_idx, dtype=int)

    if len(chosen_indices) > 0:
        biases = rng.uniform(min_bias, max_bias, size=len(chosen_indices))
        directions = rng.choice([1.0, -1.0], size=len(chosen_indices))
        local_noise = rng.normal(0.0, noise_sigma * 0.5, size=len(chosen_indices))

        for i, idx in enumerate(chosen_indices):
            attacked[idx] = attacked[idx] * (1.0 + directions[i] * float(biases[i])) + float(local_noise[i])

    global_noise = rng.normal(0.0, 0.0025, length)
    kernel = np.ones(5) / 5.0
    smooth_background = np.convolve(global_noise, kernel, mode="same")
    attacked = attacked * (1.0 + smooth_background)

    return attacked


def main():
    print("=" * 60)
    print("CREATING BIAS ATTACK SNIPPETS")
    print("=" * 60)

    normal_files = sorted(
        f for f in os.listdir(NORMAL_DIR)
        if f.startswith("n_") and f.endswith(".npy")
    )

    print(f"\nFound {len(normal_files):,} normal snippets")
    print("Creating attacked versions...")
    print("\nAttack settings:")
    print("  - random interior interval")
    print("  - per-sample multiplicative bias")
    print("  - random sign per sample")
    print("  - reduced local noise")
    print("  - smooth global background noise")

    normal_means = []
    attack_means = []
    normal_stds = []
    attack_stds = []

    for filename in tqdm(normal_files, desc="Creating bias attacks"):
        normal_path = os.path.join(NORMAL_DIR, filename)
        normal_snippet = np.load(normal_path)

        snippet_id = int(filename.split("_")[1].split(".")[0])
        attack_snippet = inject_bias_attack(normal_snippet, seed=snippet_id)

        attack_path = os.path.join(ATTACK_DIR, f"a_{snippet_id}.npy")
        np.save(attack_path, attack_snippet)

        if len(normal_means) < 200:
            normal_means.append(normal_snippet.mean())
            attack_means.append(attack_snippet.mean())
            normal_stds.append(normal_snippet.std())
            attack_stds.append(attack_snippet.std())

    print(f"\nCreated {len(normal_files):,} bias attack snippets")
    print(f"Saved to: {ATTACK_DIR}")

    normal_means = np.array(normal_means)
    attack_means = np.array(attack_means)
    normal_stds = np.array(normal_stds)
    attack_stds = np.array(attack_stds)

    print("\n" + "=" * 60)
    print("VERIFICATION (FIRST 200)")
    print("=" * 60)

    print("\nMean statistics:")
    print(f"  Normal: {normal_means.mean():.3f} V +- {normal_means.std():.3f} V")
    print(f"  Attack: {attack_means.mean():.3f} V +- {attack_means.std():.3f} V")

    avg_diff_pct = ((attack_means.mean() - normal_means.mean()) / normal_means.mean()) * 100
    print(f"  Mean difference: {avg_diff_pct:+.2f}%")

    print("\nStandard deviation statistics:")
    print(f"  Normal std: {normal_stds.mean():.4f} V +- {normal_stds.std():.4f} V")
    print(f"  Attack std: {attack_stds.mean():.4f} V +- {attack_stds.std():.4f} V")

    std_diff_pct = ((attack_stds.mean() - normal_stds.mean()) / normal_stds.mean()) * 100
    print(f"  Std difference: {std_diff_pct:+.2f}%")

    normal_range = [normal_means.min(), normal_means.max()]
    attack_range = [attack_means.min(), attack_means.max()]

    print("\nMean voltage ranges:")
    print(f"  Normal: [{normal_range[0]:.3f}, {normal_range[1]:.3f}] V")
    print(f"  Attack: [{attack_range[0]:.3f}, {attack_range[1]:.3f}] V")

    overlap_start = max(normal_range[0], attack_range[0])
    overlap_end = min(normal_range[1], attack_range[1])

    if overlap_end > overlap_start:
        total_range = max(normal_range[1], attack_range[1]) - min(normal_range[0], attack_range[0])
        overlap_pct = ((overlap_end - overlap_start) / total_range) * 100
        print(f"\nOverlap: {overlap_pct:.1f}%")

        if overlap_pct > 80:
            print("High overlap: challenging setting")
        elif overlap_pct > 60:
            print("Moderate overlap: reasonably challenging")
        else:
            print("Low overlap: may be too easy")
    else:
        print("\nNo overlap detected: may be too easy")

    print("\n" + "=" * 60)
    print("SAMPLE COMPARISONS")
    print("=" * 60)

    for sample_id in [0, 50, 100]:
        normal_sample = np.load(os.path.join(NORMAL_DIR, f"n_{sample_id}.npy"))
        attack_sample = np.load(os.path.join(ATTACK_DIR, f"a_{sample_id}.npy"))

        mean_diff_pct = ((attack_sample.mean() - normal_sample.mean()) / normal_sample.mean()) * 100
        std_diff_pct = ((attack_sample.std() - normal_sample.std()) / normal_sample.std()) * 100

        print(f"\nSnippet {sample_id}:")
        print(f"  Normal: mean={normal_sample.mean():.3f} V, std={normal_sample.std():.4f} V")
        print(f"  Attack: mean={attack_sample.mean():.3f} V, std={attack_sample.std():.4f} V")
        print(f"  Mean diff: {mean_diff_pct:+.2f}%")
        print(f"  Std diff:  {std_diff_pct:+.2f}%")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()