# AntiDeepfake Interpretable Ensemble

Interpretable speech deepfake detection based on artifact-specific expert models.

## Overview

This repository implements an interpretable framework for speech deepfake detection based on explicit synthesis artifact modeling.
Each expert model is trained to identify a predefined synthesis artifact using:
- controlled pseudo-fake generation
- artifact-oriented feature representations
- calibrated log-likelihood ratio (LLR) outputs

Expert evidence is then aggregated through:
- additive or max pooling
- learned gating network 

The pipeline is modular and designed to support the integration of new artifact experts.

---

## Method Summary

1. **Artifact-specific expert training**
   - Each expert specializes in a predefined synthesis artifact.

2. **Expert calibration**
   - Raw logits are transformed into calibrated log-likelihood ratios (LLRs).

3. **Ensemble aggregation**
   - Expert evidence is combined using summation, max fusion, or gating.

---

## Repository Structure

```text
.
├── main/                  # Main training/evaluation entrypoints
├── models/                # Expert and gating model definitions
├── utils/                 # Training, calibration, and data utilities
├── notebooks/             # Analysis and visualization notebooks
├── config/                # Configuration files
└── environment.yml        # Conda environment
```

### Main Scripts

```text
main/
├── 1__expert_training.py
├── 1a__quick_expert_eval.py
├── 2__ensemble_calibration.py
├── 3a__train_gate.py
└── 3b__eval_gate.py
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate moe_env
```

---

## Configuration

```text
config/
├── args_params.yaml
└── args_paths.yaml
```

Update paths before running experiments.

---

## Training and Evaluation Pipeline

### 1. Train artifact experts

```bash
python main/1__expert_training.py
```

### 2. Quick expert evaluation

```bash
python main/1a__quick_expert_eval.py
```

### 3. Calibrate expert outputs

```bash
python main/2__ensemble_calibration.py
```

- score calibration (KDE / CMLG)
- additive fusion
- max fusion

### 4. Train gating network (weighted fusion)

```bash
python main/3a__train_gate.py
```

### 5. Evaluate the gating network ensemble 

```bash
python main/3b__eval_gate.py
```

---

## Additional Resources

On the server, you additionally will find the following folders:

```text
checkpoints/        # Checkpoints used for the paper results (I removed all the others for ease of use)
results/            # proxy .txt files used in the notebook for plotting the results and evaluation metrics
figures/            # Figures exported from the other notebook and used in the paper
f0_trajectories/    # Cached extracted F0 trajectories

