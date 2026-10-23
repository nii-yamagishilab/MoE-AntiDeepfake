# Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores

[![Code License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.21127-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.21127)

Research code for artifact-specific speech deepfake detection by Viola Negroni, Xin Wang, Wanying Ge, Paolo Bestagini, Junichi Yamagishi, and Stefano Tubaro. See the [paper](https://arxiv.org/abs/2607.21127) and citation below.

The pipeline trains experts on bona fide speech and controlled pseudo-fake artifacts, calibrates their scores, and combines the evidence through sum/max fusion or a learned mixture-of-experts (MoE) gate. Artifact scores and gate weights support analysis of which experts contribute to a detection.

## Contents

- [Repository and supplied checkpoints](#repository-and-supplied-checkpoints)
- [Setup](#setup)
- [Configuration and data preparation](#configuration-and-data-preparation)
- [Inference with supplied experts](#inference-with-supplied-experts)
- [Train artifact experts](#train-artifact-experts)
- [Calibrate experts and fuse scores](#calibrate-experts-and-fuse-scores)
- [Train and evaluate the gate](#train-and-evaluate-the-gate)
- [Outputs and score interpretation](#outputs-and-score-interpretation)
- [Troubleshooting and implementation notes](#troubleshooting-and-implementation-notes)

## Repository and supplied checkpoints

```text
.
├── main/                       # Training, calibration, and evaluation entrypoints
├── models/                     # Expert architectures and MoE models
├── utils/                      # Data loading, artifact generation, features, metrics
├── config/
│   ├── args_params.yaml        # Hyperparameters and feature normalization statistics
│   └── args_paths.yaml         # Dataset and output paths
├── checkpoints/
│   ├── experts/                # Six supplied expert state dictionaries
│   └── MoE/                    # Supplied five-expert gate model state dictionary
├── notebooks/                  # Paper results and feature visualization notebooks
├── environment.yml            # Exported Conda environment
├── training.sh                 # Slurm example: gate training (not expert training)
├── calibration.sh              # Slurm example: calibration and ensemble scoring
├── evaluation.sh               # Slurm example: gate evaluation
├── archived.tar.gz             # Archived code; not needed for the documented entrypoints
└── LICENSE
```

The default ensemble uses the following **fixed order**:

| Order | Artifact / `mod` | Architecture | Checkpoint in `checkpoints/experts/` |
| --- | --- | --- | --- |
| 1 | F0 inconsistency / `f0_incons` | `MLP_F0_Incons` | `MIX_F0_JOINT_MLP.pth` |
| 2 | Oversmoothing / `oversmooth` | `MLP_Smooth` | `MIX_OVERSMOOTH_MLP.pth` |
| 3 | Minimum phase / `min_phase` | `LCNN` | `MIX_MIN_PHASE_LCNN.pth` |
| 4 | Energy / `energy` | `MLP_Energy` | `MIX_ENERGY_MLP.pth` |
| 5 | Tonal artifacts / `tonal` | `MLP_Checkerboard` | `MIX_TONAL_MLP.pth` |

`MIX_FILTERING_MLP.pth` is also supplied, but filtering is excluded from the default five-expert registries and gate. The supplied gate is `checkpoints/MoE/MoE_Way3_Mix_Frozen_5Exp.pth`.

Datasets, mixed-training manifests, F0 caches, calibration parameters, and experiment results are **not included**. Expert inference can use the supplied weights directly; gate inference additionally needs `all_a.npy` and `all_b.npy` (see calibration below).

## Setup

Run all commands from the repository root. Scripts add the current working directory to Python's import path and use relative data/output paths.

```bash
conda env create -f environment.yml
conda activate moe_env
python -c "import torch, torchaudio, librosa, pysptk, yaml, sklearn; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The exported environment specifies Python 3.12.12, PyTorch/torchaudio 2.10.0, NumPy 2.3.5, librosa 0.11.0, and pysptk 1.0.1. Use Python 3.12 or newer: the scripts contain f-string syntax that older Python versions cannot parse. The environment includes Linux and NVIDIA CUDA packages, as well as notebook/analysis dependencies; it is a full research environment export. If Conda tries to use the author's location, remove the final `prefix:` line from a local copy of `environment.yml` before creating the environment.

Scripts automatically select CUDA when available and otherwise use the CPU. Feature extraction also runs on the CPU; inference and training speed depend on both audio preprocessing and model execution.

## Configuration and data preparation

### Where to change settings

The entrypoints read `config/args_params.yaml`, merge `config/args_paths.yaml`, then assign experiment settings inside their `if __name__ == '__main__':` blocks. **There is no command-line argument parser.** Edit the settings in the relevant Python file before running it; assignments there override YAML values.

| Setting | Purpose / default |
| --- | --- |
| `rootpath` | Repository/cache root, default `./`; does not automatically prefix every other path |
| `save_model_folder` | Base checkpoint directory, default `checkpoints` |
| `save_results_folder` | Base output directory, default `results` |
| `seed` | Random seed, default `1234` |
| `fs`, `win_len` | Default 16,000 Hz and 3-second windows; keep consistent with feature extraction and supplied weights |
| `batch_size`, `batch_size_eval` | Training/evaluation batches, default `128` / `64` |
| `num_epochs`, `early_stopping` | Maximum `100` epochs / `20` epochs without improvement |
| `lr`, `lr_moe`, `eta_min` | Expert/gate learning rates `1e-3`; gate scheduler minimum `1e-5` |
| `wb`, `wa` | Gate detection and attribution loss weights, `1` / `0.2` |
| `mod` | Single artifact for an expert; `all` for the five-expert gate |

The `train_stats_*` sections contain feature normalization statistics. Match `train_data` to the checkpoint's training domain, not the test dataset. Supplied `MIX_*` weights use `train_data = 'mix'`. Changing this string alone does not retrain or adapt a checkpoint.

### Dataset paths and protocols

Obtain the audio and protocols separately, then replace the machine-specific paths in `config/args_paths.yaml`. Only datasets used by your chosen experiment are required.

| Dataset key | Expected inputs and behavior |
| --- | --- |
| `asv19` | ASVspoof 2019 LA protocols and train/dev/eval FLAC folders; paths are relative to `asv19.base_path` |
| `asv5` | ASVspoof 5 protocols and `flac_T`, `flac_D`, `flac_E_eval` folders |
| `sc` | SpoofCeleb CSV metadata with `attack` and `file` columns, plus split audio folders |
| `wf` | Local `protocol_full.txt` and `wav_16k/`; protocol columns are configured in YAML |
| `mlaad` | English fake WAV files under `fake_path/en/<generator>/*.wav`; current evaluation uses hard-coded ASVspoof 2019 bona fide reference paths in `utils/data_utils.py` |
| `asv15` | `CM_protocol/cm_evaluation.ndx` and WAV paths built as `<file_dir>/<speaker>/<utterance>.wav` |

ASV protocols are read with `pandas.read_csv(..., sep=' ', header=None)`, including ASVspoof 5 files whose names end in `.tsv`. Check the actual delimiter and configured zero-based column indices; adapt the reader or prepare space-separated protocol copies if your files are tab-separated. For ASVspoof 2019, the configured path, generator, and label columns are 1, 3, and 4.

Other dataset paths in the YAML are used by auxiliary/older utilities and do not imply support in every main entrypoint. Check the `test_data` branches in your selected script before changing datasets.

### Mixed-data training manifests

The default training setting is `train_data = 'mix'`, which enables `large_scale`. The active loader in `utils/data_utils.py::make_df_bonafide` reads:

```text
~/v-viola/Mix_Train.txt
~/v-viola/Mix_Dev.txt
```

These paths are hard-coded and are not controlled by `sc.real_T_dir` or `sc.real_D_dir`. Create your own train/dev manifests and **edit those two reads to point to them**. Each file must contain one bona fide audio path per line, with no header; absolute paths are easiest to use:

```text
/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac
/data/ASVspoof5/flac_T/example.flac
/data/SpoofCeleb/flac/train/example.flac
```

Use real existing filenames and keep training and validation manifests disjoint. The intended mix combines ASVspoof 2019, ASVspoof 5, and VoxCeleb/SpoofCeleb bona fide speech. The commented manifest-construction code is not executed. Pseudo calibration with `train_data = 'mix'` also needs these manifests.

For a smaller single-corpus experiment, set `train_data = 'asv19'` in the training script. It sets `large_scale = False`: the loader combines bona fide train/dev audio and makes a random 80/20 split with `random_state=42`. This is an utterance split, not a speaker-disjoint split. `asv5` also enables 15% subsampling; some artifact normalization statistics are missing for single-corpus settings (see troubleshooting).

### Audio preprocessing and F0 cache

Audio is loaded as mono and resampled to 16 kHz. Training uses trimming and peak normalization. Non-F0 features use fixed windows, with random crops for training and a middle-window selection for evaluation; short audio is repeated. F0 features use a separate trajectory extraction/cache path.

The training, sanity-check/calibration, and evaluation dataloaders automatically create `<rootpath>/f0_trajectories/` and its dataset subfolders (`f0_asv19`, `f0_asv5`, `f0_sc`, `f0_mlaad`, `f0_mailabs`, `f0_asv15`, and `f0_wf`) when initialized. Existing folders and cached files are preserved; no manual `mkdir` step is needed. Ensure `rootpath` is writable. Dataset detection for F0 depends on case-sensitive substrings in audio paths: `ASVspoof2019`, `ASVspoof5`, `SpoofCeleb`, `MLAAD`, `M-AILABS`, `ASVspoof2015`, or `WaveFake`. For other directory names, update the folder mapping in `utils/dataloader.py`. Cache filenames hash the audio path; moving data changes the cache key. Remove affected caches after changing audio or F0 preprocessing.

## Inference with supplied experts

The quickest supplied demonstration evaluates five experts on selected ASVspoof 2019 attacks without retraining or calibration:

1. Activate the environment, configure `asv19.base_path` and its protocols/audio folders, and ensure `rootpath` is writable for the automatic F0 cache.
2. Keep `train_data = 'mix'` and `test_data = 'asv19'` in `main/example_1_1a__quick_expert_eval.py`.
3. Run:

```bash
python main/example_1_1a__quick_expert_eval.py
```

The demonstration evaluates F0 and energy on A04, tonal artifacts on A01, and minimum phase and oversmoothing on A03. It prints AUC/EER and saves `file_IDs.npy`, `scores.npy`, and `labels.npy` beneath:

```text
results/experts/asv19/experts_trained_on_mix/<mod>/TargetCheck/<attack>/
```

For an individual expert, edit `main/1a__quick_expert_eval.py`:

```python
config['mod'] = 'tonal'
config['train_data'] = 'mix'
config['test_data'] = 'asv19'
config['sanity_check'] = False
config['target_check'] = True
config['eval_all'] = False
config['model'] = 'MLP'
model_classes = [MLP_Checkerboard]
# Use MIX_TONAL_MLP.pth in model_names.
```

Then run:

```bash
python main/1a__quick_expert_eval.py
```

This script defaults to tonal/A01 evaluation and **prints metrics only**: its three `np.save` calls are commented out. Enable them if you need saved scores. Match `mod`, model class, and checkpoint using the table above. `eval_all = True` pools the attacks in `selected`; it does not automatically select every attack in the corpus. Edit the dataset-specific `selected` assignment to change the attack subset. With `eval_all = False`, attacks are evaluated separately.

These entrypoints perform protocol-based batch inference and metrics. There is no ready-made `--audio file.wav` interface. To score custom files, adapt the data-loading path (for example, construct `LoadEvaluationData` with file paths, labels, and the same config/model order). The existing evaluation helpers expect labels for AUC/EER, and F0 needs the dataset mapping described above.

## Train artifact experts

Edit the settings near the bottom of `main/1__expert_training.py`. The checked-in defaults train a **filtering MLP on mixed data**:

```python
config['train_data'] = 'mix'
config['model'] = 'mlp'
config['mod'] = 'filtering'
# The script derives model_name from these settings, with a _v2.pth suffix.
```

Choose one artifact per run:

| `mod` | `model` | Model class selected by the script |
| --- | --- | --- |
| `f0_incons` or `flat_pitch` | `mlp` | `MLP_F0_Incons` |
| `oversmooth` | `mlp` | `MLP_Smooth` |
| `min_phase` | `lcnn` | `LCNN` |
| `energy` | `mlp` | `MLP_Energy` |
| `tonal` or `filtering` | `mlp` | `MLP_Checkerboard` |

```bash
python main/1__expert_training.py
```

The loader duplicates bona fide paths to generate real/pseudo-fake examples and applies the selected artifact during feature preparation. Expert optimization uses AdamW, binary cross-entropy with logits, and a validation-loss scheduler. The best validation-loss state dictionary is saved under `checkpoints/experts/<model_name>`; the run stops after the configured patience or maximum epochs. There is no checkpoint-resume option in this entrypoint.

Repeat for the five default ensemble artifacts. Newly trained filenames differ from supplied names: for example, F0 training generates `MIX_F0_INCONS_MLP_v2.pth`, whereas ensemble scripts expect `MIX_F0_JOINT_MLP.pth`. Update downstream checkpoint lists explicitly. Preserve architecture tokens such as `MLP` and `LCNN`, which the ensemble loader uses to identify model types. Use distinct names to preserve existing weights.

## Calibrate experts and fuse scores

`main/2__ensemble_calibration.py` fits calibration parameters **and then evaluates the experts**. It requires both calibration inputs and the configured test dataset. Edit its settings and checkpoint list before running:

```python
config['calibration_type'] = 'CMLG'  # or 'KDE'; checked-in default is KDE
config['train_data'] = 'mix'
config['calibration_data'] = 'pseudo'
config['test_data'] = 'asv19'
config['pool_calibration_set'] = False
config['calib_class_samples'] = 1000
config['all_manipulations'] = [
    'f0_incons', 'oversmooth', 'min_phase', 'energy', 'tonal'
]
```

```bash
python main/2__ensemble_calibration.py
```

- **CMLG** fits an affine log-likelihood ratio, `LLR = a * raw_logit + b`, using class means and a shared variance.
- **KDE** fits Platt scaling followed by class-conditional kernel density estimates for LLR scoring.
- **Pseudo calibration** uses sampled bona fide validation audio and manipulated counterparts. Keep `pool_calibration_set = False` explicitly for this mode.
- Real calibration data branches include `asv19`, `asv5`, `asv15`, and `sc`. ASVspoof 5 and SpoofCeleb require pooled calibration in the current loader. Choose a sample count available for every selected class/attack.

For the CMLG configuration above, outputs go to:

```text
results/Ensemble/asv19/experts_trained_on_mix/CMLG/5_experts/Cset_PSEUDO_1000/
```

CMLG saves `all_a.npy`, `all_b.npy`, `all_mf.npy`, `all_mr.npy`, and `all_v.npy`. Both calibration methods save aligned `scores_per_expert.npy`, `raw_logits_per_expert.npy`, `labels.npy`, `attack_ids.npy`, and `filenames.npy`. KDE additionally saves density objects through NumPy with filenames `all_H1.pkl.npy` and `all_H0.pkl.npy`; these are not ordinary pickle files.

The script saves per-expert evidence; it does **not** automatically print or save sum/max fusion metrics. Compute these from the saved LLR matrix, whose axes are `(experts, samples)`:

```python
from pathlib import Path
import numpy as np
from utils.utils import compute_metrics_v2

folder = Path('results/Ensemble/asv19/experts_trained_on_mix/CMLG/5_experts/Cset_PSEUDO_1000')
llrs = np.load(folder / 'scores_per_expert.npy')
labels = np.load(folder / 'labels.npy')
for name, scores in [('sum', llrs.sum(axis=0)), ('max', llrs.max(axis=0))]:
    auc, eer, _ = compute_metrics_v2(labels, scores)
    print(f'{name}: AUC={100 * auc:.2f}%, EER={100 * eer:.2f}%')
```

## Train and evaluate the gate

### Train a new gate

First train or select the five expert checkpoints and run **CMLG** calibration. In `main/3a__train_gate.py`, edit `setup_moe_model` so its checkpoint paths match your experts and its scaling path matches your generated calibration directory:

```python
config['train_data'] = 'mix'
config['scaling_params'] = (
    'Ensemble/asv19/experts_trained_on_mix/CMLG/5_experts/Cset_PSEUDO_1000'
)
```

`scaling_params` is relative to `save_results_folder`. The checked-in value points to `Cset_ASV19_pooled_50_dev_eval`, which is not supplied and is not the default calibration output. Replace it in **both** gate scripts. Keep YAML `mod: 'all'` for gate runs and keep expert, manipulation, and calibration parameter ordering identical.

```bash
python main/3a__train_gate.py
```

The script defaults to freezing experts and training `MOE_Way3` on mixed bona fide/pseudo-fake examples with detection and artifact-attribution losses. It uses AdamW and a cosine learning-rate schedule, selecting the best checkpoint by validation EER. The default output is `checkpoints/MoE/MoE_Way3_Mix_Frozen_5Exp.pth`, which **overwrites the supplied checkpoint on improvement**. Change `model_name` in the final `setup_moe_model(...)` call to preserve it.

### Evaluate a gate checkpoint

In `main/3b__eval_gate.py`, set `ckpt_moe`, the expert list, `test_data`, `selected`, and `scaling_params` to match the trained model. Keep `train_data = 'mix'` for the supplied model. The evaluator constructs the expert architectures and loads their weights from the full MoE state dictionary; its expert filenames determine architecture ordering.

```bash
python main/3b__eval_gate.py
```

The default pools its selected ASVspoof 2019 attacks and writes:

```text
results/MoE/asv19/TargetCheck/ALL/MoE_Way3_Mix_Frozen_5Exp/
```

Set `eval_all = False` for per-attack subdirectories. The script prints AUC/EER and saves `scores.txt`, `gating_all.npy`, `weighted_all.npy`, `expert_all.npy`, and global/real/fake means. Existing files in the same evaluation directory are replaced.

**Supplied-gate reproducibility:** `MOE_Way3` stores `a_params` and `b_params` as ordinary attributes, not checkpoint buffers. The supplied `.pth` therefore does not include these required calibration arrays. Generating new CMLG parameters allows evaluation with that calibration, but does not establish exact reproduction of the supplied gate's original experiment. Preserve the exact calibration arrays alongside any newly trained gate.

### Slurm examples

The `.sh` files are scheduler templates with site-specific GPU requests, output paths, and Conda initialization. Adjust those for your cluster, create the log directory, and submit from the repository root:

```bash
mkdir -p txt
sbatch calibration.sh
# After calibration completes and scaling_params is updated:
sbatch training.sh
# After gate training completes:
sbatch evaluation.sh
```

Submit each stage only after its prerequisite finishes successfully. `training.sh` runs gate training; run `main/1__expert_training.py` separately to train experts. A local run can use the Python commands directly.

## Outputs and score interpretation

- Labels are `0 = bona fide` and `1 = fake`; higher scores indicate stronger fake evidence. Invalid examples can receive label `-1` and are removed by evaluation helpers.
- Individual experts return raw scores/logits. Calibrated ensemble arrays contain LLRs; they are not probabilities. A zero LLR represents equal evidence under the fitted class models.
- The active `MOE_Way3` implementation applies affine calibration, **clamps to `[1e-6, 1 - 1e-6]`**, computes softmax gate weights, and sums weighted scores. These outputs should not be described as an unbounded fused LLR or assumed to be calibrated posterior probabilities.
- `scores.txt` contains whitespace-separated utterance ID, score, and label. Scores may be formatted as one-element NumPy arrays (for example `[0.42]`), so strip brackets when parsing. `expert_all.npy` stores raw expert outputs; `weighted_all.npy` stores contributions after affine scaling, clipping, and gating.
- AUC and EER are printed as percentages; `compute_metrics_v2` returns fractions. Its third return value uses a zero decision threshold, which is not an appropriate default for the clipped gate score. Choose a deployment threshold on held-out validation data.
- F0 extraction failures can reduce the scored subset. Calibration evaluation aligns all experts to the valid IDs of the first (F0) expert. Compare methods on the same retained samples.

For analysis, launch `jupyter lab` and open `notebooks/plot_paper_results.ipynb`. Other notebooks cover ensemble analysis and feature examples. Notebook cells contain experiment-specific paths and may expect result files that are not bundled; update them to your generated outputs.

## Troubleshooting and implementation notes

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: utils` | Run from the repository root in `moe_env`. |
| Syntax error inside an f-string | Activate Python 3.12 or newer. |
| Missing `Mix_Train.txt` / `Mix_Dev.txt` | Update the two hard-coded manifest reads in `make_df_bonafide`; YAML dataset paths do not replace them. |
| Missing `all_a.npy` / `all_b.npy` | Run CMLG calibration, then update `scaling_params` in both gate scripts to the actual directory. |
| F0 cache write failure / unrecognized dataset | Cache directories are created automatically; check write permissions under `rootpath` and case-sensitive dataset names in audio paths. |
| Missing `mean_f0_joint` or other statistics | `f0_incons` requires `mean_f0_joint`/`std_f0_joint`, currently supplied only for `mix`; ASVspoof 5 also lacks energy/oversmooth statistics. Compute the appropriate training statistics before using those combinations. |
| Protocol columns missing or labels empty | Check delimiter, configured column indices, and exact `bonafide`/`spoof` strings. |
| Sampling error: requested sample exceeds population | Reduce `calib_class_samples`; ASVspoof 5 evaluation samples 7,760 fake files per selected attack, SpoofCeleb samples 7,963. Adapt those counts for smaller datasets. |
| Wrong model shapes or mismatched expert features | Preserve the five-expert order in `MODEL_REGISTRY`, `MANIPULATIONS_REGISTRY`, checkpoint lists, and calibration arrays. Reordering MLP filenames alone does not change the registry mapping. |
| Empty/invalid audio | Check inputs before running; completely empty audio is not handled safely by the window-repetition helper. |

Evaluation here follows the repository's research subsets. ASVspoof 2019 uses selected fake attacks from **dev + eval**, with bona fide speech from eval. ASVspoof 5 pools selected fake samples from train/dev/eval, removes post-processed eval samples, and samples per attack. SpoofCeleb similarly pools fake samples across splits. `eval_all` pools the configured attack list; these defaults are not an untouched official evaluation-only benchmark. Review `utils/data_utils.py::load_evaluation_data` and keep calibration/training/test partitions separate when designing a new experiment.

Adding a new expert requires more than adding a checkpoint: implement its feature/manipulation and normalization, update model selection, extend both registries, and keep all expert lists and calibration arrays in sync. Changing the number of experts also changes the gate shape and requires a compatible gate checkpoint or retraining.

## Acknowledgments
This work is partially supported by JST PRESTO Grant (JPMJPR23P9). This project was partially carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

The code in this repository is released under the MIT license. See [`LICENSE`](LICENSE).

## Citation

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
