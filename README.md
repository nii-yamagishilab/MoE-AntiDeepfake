# AntiDeepfake Interpretable Ensemble


```bibtex
Interpretable speech deepfake detection based on artifact-specific expert models

Proc. ACM MM DFF-Workshop, 2026
```
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
├── checkpoint/            # Checkpoints for each experiment
├── environment.yml        # Conda environment
├── *.sh                   # Example scripts to train, calibrate, and evaluate
└── archived.tar.gz        # Archived code for MOE
```

### Main Scripts

```text
main/
├── 1__expert_training.py              # training expert
├── 1a__quick_expert_eval.py           # evaluation using an expert
├── example_1_1a__quick_expert_eval.py # example to run five experts
├── 2__ensemble_calibration.py         # calibration
├── 3a__train_gate.py                  # for MOE, training gate
└── 3b__eval_gate.py                   # for MOE, evaluation
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
├── args_params.yaml     # Configuration of features
└── args_paths.yaml      # Configuration of paths
```

Update paths before running experiments:
* `rootpath`: the root path of the experiment
* `save_model_folder`: folder of checkpoints (if available, they are loaded for evaluation)
* `save_results_folder`: folder to save the results
* other paths are for dataset. Please change them to yours


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

A modified version for demonstration is in `main/example_1_1a__quick_expert_eval.py`. It uses the pre-trained checkpoints and get the EER/AUC on target attacks (Table 2 of the paper). To run it
* change `config/args_paths.yaml` to point to the current folder and correct paths to the data in your environment
* run `python main/example_1_1a__quick_expert_eval.py`
* results are saved to `results/experts`
* `notebooks/plot_paper_results.ipynb` shows how to load the files in `results/experts` and compute EER/AUC

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
results/            # proxy files used in the notebook for plotting the results and evaluation metrics
figures/            # Figures exported from the other notebook and used in the paper
f0_trajectories/    # Cached extracted F0 trajectories

