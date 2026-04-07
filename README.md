# Constraint-Driven Warm-Freeze for Efficient Transfer-Learning in Photovoltaic-Systems
Constraint-Driven Warm-Freeze (CDWF): a budget-aware transfer learning method for PV cyberattack detection. It allocates training to high-impact layers and uses LoRA for efficiency, achieving near full fine-tuning performance with up to 120× fewer trainable parameters.

This repository contains:
- Novel PV dataset generator
- CIFAR-10 / CIFAR-100 experiments
- PV cybersecurity experients (drift, spike)

Dataset can be found on:
https://www.kaggle.com/datasets/yasmeenfozi/pv-dataset

## Project Structure

```
PV-Generator/
├── (data generation scripts)

Cifar_10/
├── cifar10_LoRA-baseline.ipynb
├── cifar10_full-ft.ipynb
├── cifar10_resnet50_CDWF.ipynb

Cifar_100/
├── cifar100_LoRA-baseline.ipynb
├── cifar100_full-ft.ipynb
├── cifar100_resnet50_CDWF.ipynb

Drift/
├── pv-drift_LoRA-baseline.ipynb
├── pv-drift_full-ft.ipynb
├── pv-drift_resnet1D_CDWF.ipynb
├──best_resnet1d_basicblock_bias.pth

Spike/
├── pv-spike_LoRA-baseline.ipynb
├── pv-spike_full-ft.ipynb
├── pv-spike_resnet1D_CDWF.ipynb
├──best_resnet1d_basicblock_bias.pth
```

## PV Data Generation

### Synthetic PV data is generated using the PV-Generator module.

#### Step 1 — Generate Normal Signals
(You can create different normal signals for each attack type)

```python
python 1_generate_normal_snippets.py
```
- Simulates PV voltage signals
- Splits into fixed-length windows (10 seconds)
- Saves as .npy files

#### Step 2 — Generate Attacks

Run each attack script separately.

Bias

```python
python 2_generate_bias_attacks.py
```
- Per-sample multiplicative perturbations
- Subtle, variance-driven changes

Drift

```python
python 2_generate_drift_attacks.py
```
- Gradual temporal drift within a random interval
- Preserves signal edges

Spike

```python
python 2_generate_spike_attacks.py
```
- Localized spikes within a short interval

### Data Format

Each sample is stored as:

```python
n_0.npy   # normal
a_0.npy   # attack
```
- Paired samples share the same index

#### Notes:

All attack types follow consistent rules:
- Random interior interval
- Edge preservation
- Controlled noise injection


------------------------------------------------------------

1. This project includes three methods:
- Full_FT: full fine-tuning baseline
- LoRA: parameter-efficient baseline using low-rank adaptation
- CDWF: constraint-driven warm-freeze (proposed method)



2. The main focus of this work is the params constraint, i.e., limiting the percentage of trainable parameters while maintaining performance.

3. Full_FT is used as the reference baseline for accuracy, parameters.

4. LoRA can be run with multiple ranks (e.g., 1, 2, 4, 8, 16), where the rank controls the trade-off between efficiency and performance.

5. CDWF automatically selects which parts of the model to train or adapt (via LoRA) to satisfy the chosen constraint.

6. All methods should use the same data splits, seeds, and training setup for fair comparison.

Phase 2 of this work is complete and addresses most limitations identified in the initial code and paper. To be published soon...
