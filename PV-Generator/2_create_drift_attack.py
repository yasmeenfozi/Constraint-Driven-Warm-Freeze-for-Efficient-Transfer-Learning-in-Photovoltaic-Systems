"""Create drift-attacked versions of all normal snippets."""

import os
import numpy as np
from tqdm import tqdm


NORMAL_DIR = "/normal_vs_drift/data/normal"
ATTACK_DIR = "/normal_vs_drift/data/attack"
os.makedirs(ATTACK_DIR, exist_ok=True)


def inject_drift_attack(
    window: np.ndarray,
    seed: int | None = None,
    min_mag: float = 0.02,
    max_mag: float = 0.03,
    noise_sigma: float = 0.0015,
) -> np.ndarray:
    """
    Apply a drift attack over a random interior interval of the signal.

    The attack:
    - protects the first and last parts of the window
    - selects one random interior interval
    - applies a linear multiplicative drift only within that interval
    - adds local noise only to the attacked region
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

    if end_idx > start_idx + 1:
        interval_len = end_idx - start_idx
        final_mag = float(rng.uniform(min_mag, max_mag))
        direction = float(rng.choice([1.0, -1.0]))

        ramp = np.linspace(0.0, final_mag, interval_len, dtype=np.float64)
        scale = 1.0 + direction * ramp

        attacked[start_idx:end_idx] = attacked[start_idx:end_idx] * scale

        local_noise = rng.normal(0.0, noise_sigma, size=interval_len)
        attacked[start_idx:end_idx] = attacked[start_idx:end_idx] + local_noise

    global_noise = rng.normal(0.0, 0.0025, length)
    kernel = np.ones(5) / 5.0
    smooth_background = np.convolve(global_noise, kernel, mode="same")
    attacked = attacked * (1.0 + smooth_background)

    return attacked


def main():
    print("=" * 60)
    print("CREATING DRIFT ATTACK SNIPPETS")
    print("=" * 60)

    normal_files = sorted(
        f for f in os.listdir(NORMAL_DIR)
        if f.startswith("n_") and f.endswith(".npy")
    )

    print(f"\nFound {len(normal_files):,} normal snippets")
    print("Creating attacked versions...")
    print("\nAttack settings:")
    print("  - one random interior interval")
    print("  - linear drift with random direction")
    print("  - local noise on attacked indices only")
    print("  - smooth global background noise")

    normal_means = []
    attack_means = []

    for filename in tqdm(normal_files, desc="Creating drift attacks"):
        normal_signal = np.load(os.path.join(NORMAL_DIR, filename))
        snippet_id = int(filename.split("_")[1].split(".")[0])

        attacked_signal = inject_drift_attack(
            normal_signal,
            seed=snippet_id + 10_000_000,
        )

        np.save(os.path.join(ATTACK_DIR, f"a_{snippet_id}.npy"), attacked_signal)

        if len(normal_means) < 200:
            normal_means.append(normal_signal.mean())
            attack_means.append(attacked_signal.mean())

    print(f"\nCreated {len(normal_files):,} drift attack snippets")
    print(f"Saved to: {ATTACK_DIR}")

    normal_means = np.array(normal_means)
    attack_means = np.array(attack_means)

    print("\n" + "=" * 60)
    print("VERIFICATION (FIRST 200)")
    print("=" * 60)

    print(f"\nNormal mean: {normal_means.mean():.3f} V +- {normal_means.std():.3f} V")
    print(f"Attack mean: {attack_means.mean():.3f} V +- {attack_means.std():.3f} V")

    avg_diff_pct = ((attack_means.mean() - normal_means.mean()) / max(1e-12, normal_means.mean())) * 100
    print(f"Average difference: {avg_diff_pct:+.2f}%")

    normal_range = [normal_means.min(), normal_means.max()]
    attack_range = [attack_means.min(), attack_means.max()]

    print("\nMean voltage ranges:")
    print(f"  Normal: [{normal_range[0]:.3f}, {normal_range[1]:.3f}] V")
    print(f"  Attack: [{attack_range[0]:.3f}, {attack_range[1]:.3f}] V")

    overlap_start = max(normal_range[0], attack_range[0])
    overlap_end = min(normal_range[1], attack_range[1])

    if overlap_end > overlap_start:
        total_range = max(normal_range[1], attack_range[1]) - min(normal_range[0], attack_range[0])
        overlap_pct = ((overlap_end - overlap_start) / max(1e-12, total_range)) * 100
        print(f"\nOverlap: {overlap_pct:.1f}%")

        if overlap_pct > 80:
            print("High overlap: very challenging setting")
        elif overlap_pct > 60:
            print("Moderate overlap: challenging setting")
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

        diff_pct = ((attack_sample.mean() - normal_sample.mean()) / max(1e-12, normal_sample.mean())) * 100

        print(f"\nSnippet {sample_id}:")
        print(f"  Normal: {normal_sample.mean():.3f} V +- {normal_sample.std():.3f} V")
        print(f"  Attack: {attack_sample.mean():.3f} V +- {attack_sample.std():.3f} V")
        print(f"  Mean diff: {diff_pct:+.2f}%")
        print(f"  First 10 normal: {normal_sample[:10]}")
        print(f"  First 10 attack: {attack_sample[:10]}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()