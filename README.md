<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores</p>
</div>

<div align="center" style="line-height: 1;">
  <a href="./LICENSE"><img alt="Code License"
    src="https://img.shields.io/badge/license-MIT-blue"/></a>
  <a href="https://arxiv.org/abs/2607.21127"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-2607.21127-b31b1b?logo=arxiv&logoColor=white"/></a>
</div>

# Introduction

This is the code repository for paper accepted below

```bibtex
Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores
Viola Negroni, Xin Wang, Wanying Ge, Paolo Bestagini, Junichi Yamagishi, Stefano Tubaro
Proc. ACM Multimedia, 2nd Deepfake Forensics Workshop: Detection, Attribution, Recognition, and Adversarial Challenges in the Era of AI-Generated Media, 2026
```

[Arxiv link](https://arxiv.org/abs/2607.21127)


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


## 6. Acknowledgments
It is  partially supported by JST PRESTO Grant (JPMJPR23P9). This project was partially carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

The code in this repository is released under the MIT license. See [`LICENSE`](LICENSE).

## 7. Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{negroni2026acmmmw,
  title={Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores},
  author={Negroni, Viola and Wang, Xin and Ge, Wanying and Bestagini, Paolo and Yamagishi, Junichi and Tubaro, Stefano},
  booktitle={Proc. ACM Multimedia, 2nd Deepfake Forensics Workshop: Detection, Attribution, Recognition, and Adversarial Challenges in the Era of AI-Generated Media},
  year={2026}
}
```
---
