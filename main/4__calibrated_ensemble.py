import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import os
sys.path.insert(1, os.getcwd())

from utils.data_utils import load_calibration_data, load_evaluation_data
from pathlib import Path
from utils.training_utils import expert_loop
from utils.utils import *
from utils.model_utils import MODEL_REGISTRY, get_model_type_from_ckpt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
import joblib


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def platt_scaling(val_loader, expert):
    logits_list, label_list = expert_loop(val_loader, expert, device)
    logreg = LogisticRegression()
    y = np.array(label_list)
    X = np.array(logits_list).reshape(-1, 1)  # shape (N, 1)

    logreg.fit(X, y)

    a = logreg.coef_[0][0]
    b = logreg.intercept_[0]

    return a, b

def calibrate_expert(calib_loader, expert, a, b):
    logits_list, label_list = expert_loop(calib_loader, expert, device)

    logits = np.array(logits_list)
    labels = np.array(label_list)

    probs = 1 / (1 + np.exp(-(a * logits + b)))
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    # probs = logits

    p_H1 = probs[labels == 1].reshape(-1, 1)
    p_H0 = probs[labels == 0].reshape(-1, 1)

    bandwidth = 0.1  # da capire
    kde_H1 = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_H1.fit(p_H1)
    kde_H0 = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_H0.fit(p_H0)

    return kde_H1, kde_H0

def run_calibration(experts, models_list, config):
    # Scale logits, fit KDEs for each expert

    config['pool'] = True
    calib_loader = load_calibration_data(models_list, config)
    config['pool'] = False

    all_a = []
    all_b = []
    all_kde_H1 = []
    all_kde_H0 = []
    mods = ['flat_pitch', 'f0_incons', 'oversmooth', 'min_phase', 'energy']  # mind the order!!!

    for i, expert in enumerate(experts):
        print(f"CALIBRATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)

        val_loader = load_calibration_data(mods[i], config)

        val_loader.dataset.mod = mods[i]
        calib_loader.dataset.mod = mods[i]

        a, b = platt_scaling(val_loader, expert)
        kde_H1, kde_H0 = calibrate_expert(calib_loader, expert, a, b)

        all_a.append(a)
        all_b.append(b)
        all_kde_H1.append(kde_H1)
        all_kde_H0.append(kde_H0)

        print()

    return all_a, all_b, all_kde_H1, all_kde_H0

def compute_log_lr(eval_loader, expert, a, b, kde_H1, kde_H0):
    logits_list, label_list, ids = expert_loop(eval_loader, expert, device, return_id=True)

    logits = np.array(logits_list)
    probs = 1 / (1 + np.exp(-(a * logits + b)))
    probs = np.clip(probs, 1e-6, 1 - 1e-6).reshape(-1, 1)
    # probs = logits.reshape(-1, 1)

    log_p_H1 = kde_H1.score_samples(probs)
    log_p_H0 = kde_H0.score_samples(probs)

    return log_p_H1 - log_p_H0, label_list, ids, logits, probs

def run_evaluation(experts, eval_loader, all_a, all_b, all_H1, all_H0, config):

    # Compute likelihood ratios for each expert

    mods = ['flat_pitch', 'f0_incons', 'oversmooth', 'min_phase', 'energy']  # mind the order!!!
    all_results = []

    for i, expert in enumerate(experts):
        print(f"EVALUATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)
        eval_loader.dataset.mod = mods[i]

        a = all_a[i]
        b = all_b[i]
        kde_H1 = all_H1[i]
        kde_H0 = all_H0[i]

        logLR, labels, ids, raw_logits, scaled_logits = compute_log_lr(eval_loader, expert, a, b, kde_H1, kde_H0)

        all_results.append({
            "ids": np.array(ids),
            "logLR": np.array(logLR),
            "labels": np.array(labels),
            "raw_logits": np.array(raw_logits),
            "scaled_logits": np.array(scaled_logits)
        })

        print()

    print(f"\nComputing ensemble scores...")

    # Keep only samples valid for all F0-based experts (here first one must be F0-based!)
    common_ids = all_results[0]["ids"]
    aligned_logLR = []
    aligned_raw_logits = []
    aligned_scaled_logits = []
    labels_ref = None

    for res in all_results:
        ids = res["ids"]
        logLR = res["logLR"]
        labels = res["labels"]
        raw_logits = res["raw_logits"]
        scaled_logits = res["scaled_logits"]

        id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        idxs = [id_to_idx[id_] for id_ in common_ids]

        aligned_logLR.append(logLR[idxs])
        aligned_raw_logits.append(raw_logits[idxs])
        aligned_scaled_logits.append(scaled_logits[idxs])

        if labels_ref is None:
            labels_ref = labels[idxs]

    all_logLR = np.stack(aligned_logLR)  # shape: (n_experts, N)
    all_raw_logits = np.stack(aligned_raw_logits)
    all_scaled_logits = np.stack(aligned_scaled_logits)

    # Compute ensemble scores
    ensemble_logLR = np.sum(all_logLR, axis=0)
    # preds = (ensemble_logLR > 0).astype(int)  # binary predictions

    # Print and save results
    scores = np.array(ensemble_logLR)
    labels = np.array(labels_ref)
    rocauc, eer = compute_metrics_v2(labels, scores)
    print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n\n")

    np.save(f"{config['save_path']}/scores_per_expert.npy", all_logLR)
    np.save(f"{config['save_path']}/scaled_logits_per_expert.npy", all_scaled_logits)
    np.save(f"{config['save_path']}/raw_logits_per_expert.npy", all_raw_logits)

    np.save(f"{config['save_path']}/scores.npy", scores)
    np.save(f"{config['save_path']}/labels.npy", labels)


def test_ensemble(ckpts_experts, config):

    # Load expert models
    experts = []
    models_list = []
    model_registry = MODEL_REGISTRY.copy()

    for ckpt in ckpts_experts:
        model_type = get_model_type_from_ckpt(ckpt)
        for name, mod in model_registry:
            if name == model_type:
                model_class = mod
                model_registry.remove((name, model_class))
                break
        model = model_class(return_emb=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        experts.append(model)
        models_list.append(model_type)

    # Load necessary data
    # config['merge'] = True
    # _, val_loader = load_training_data(models_list, config)
    # calib_loader = load_evaluation_data(models_list, config)
    # config['target_check'] = True

    # Calibrate & evaluate
    Path(config['save_folder']).mkdir(parents=True, exist_ok=True)
    files = {name: f"{config['save_folder']}/{name}{'.pkl' if 'H' in name else '.npy'}"
             for name in ['all_a', 'all_b', 'all_H1', 'all_H0']}

    if all(os.path.exists(f) for f in files.values()):
        all_a = np.load(files['all_a'], allow_pickle=True)
        all_b = np.load(files['all_b'], allow_pickle=True)
        all_H1 = joblib.load(files['all_H1'])
        all_H0 = joblib.load(files['all_H0'])
    else:
        all_a, all_b, all_H1, all_H0 = run_calibration(experts, models_list, config)
        np.save(files['all_a'], all_a)
        np.save(files['all_b'], all_b)
        joblib.dump(all_H1, files['all_H1'])
        joblib.dump(all_H0, files['all_H0'])

    config['target_check'] = True

    if config['eval_overall']:
        config['save_path'] = os.path.join(config['save_folder'], "ALL")

        Path(config['save_path']).mkdir(parents=True, exist_ok=True)
        print(f"Result path: {config['save_path']} \n")

        eval_loader = load_evaluation_data(models_list, config)
        run_evaluation(experts, eval_loader, all_a, all_b, all_H1, all_H0, config)

    else:
        targets = config['selected']
        for target in targets:
            config['selected'] = [target]
            config['save_path'] = os.path.join(config['save_folder'], target)

            Path(config['save_path']).mkdir(parents=True, exist_ok=True)
            print(f"Result path: {config['save_path']} \n")

            eval_loader = load_evaluation_data(models_list, config)
            run_evaluation(experts, eval_loader, all_a, all_b, all_H1, all_H0, config)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")
    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])

    ckpt_exp_path = os.path.join(config['save_model_folder'], 'experts')
    save_path = os.path.join(config['save_results_folder'], 'Ensemble')

    # --------------------- PLEASE DEFINE ----------------------#

    config['train_data'] = 'asv19'
    config['test_data'] = 'asv5'
    config['eval_overall'] = True

    # ------------------------- Thanks -------------------------#

    print("\nTEST CALIBRATED ENSEMBLE\n")

    if config['train_data'] == 'asv19':
        ckpts_experts = [
            os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'F0_INCONS_MLP.pth'),
            os.path.join(ckpt_exp_path, 'OVERSMOOTH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN.pth'),
            os.path.join(ckpt_exp_path, 'ENERGY_MLP.pth')
        ]
    else:
        ckpts_experts = [
            os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP_ASV5.pth'),
            os.path.join(ckpt_exp_path, 'F0_INCONS_MLP_ASV5.pth'),
            os.path.join(ckpt_exp_path, 'OVERSMOOTH_GRU_ASV5.pth'),
            os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN_ASV5.pth'),
            os.path.join(ckpt_exp_path, 'ENERGY_LCNN_ASV5.pth')
        ]

    if config['test_data'] == 'asv19':
        config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
    elif config['test_data'] == 'asv5':
        config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A14']
    else:
        print("Test dataset not supported.\n")
        sys.exit()
    config['save_folder'] = os.path.join(save_path,
                                         config['test_data'],
                                         f'experts_trained_on_{config["train_data"]}',
                                         'NEW')
    config['target_check'] = False
    config['sanity_check'] = False
    if config['train_data'] == 'asv5':
        config['reduce_set'] = True
    else:
        config['reduce_set'] = False

    test_ensemble(ckpts_experts, config)




