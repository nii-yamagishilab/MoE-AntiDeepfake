import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from utils.data_utils import load_calibration_data
from utils.training_utils import expert_loop_calibration


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def platt_scaling(val_loader, expert):
    logits_list, label_list, _, _ = expert_loop_calibration(val_loader, expert, device)
    logreg = LogisticRegression()
    y = np.array(label_list)
    X = np.array(logits_list).reshape(-1, 1)  # shape (N, 1)

    logreg.fit(X, y)

    a = logreg.coef_[0][0]
    b = logreg.intercept_[0]

    return a, b

def fit_kde(calib_loader, expert, a, b):
    logits_list, label_list, _, _ = expert_loop_calibration(calib_loader, expert, device)

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

def run_calibration_KDE(experts, models_list, config):
    # Scale logits, fit KDEs for each expert

    if config['pool_calibration_set']:
        calib_loader = load_calibration_data(models_list, config)

    all_a = []
    all_b = []
    all_kde_H1 = []
    all_kde_H0 = []

    for i, expert in enumerate(experts):
        print(f"CALIBRATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)

        if not config['pool_calibration_set']:
            calib_loader = load_calibration_data(config["all_manipulations"][i], config)

        calib_loader.dataset.mod = config["all_manipulations"][i]

        a, b = platt_scaling(calib_loader, expert)
        kde_H1, kde_H0 = fit_kde(calib_loader, expert, a, b)

        all_a.append(a)
        all_b.append(b)
        all_kde_H1.append(kde_H1)
        all_kde_H0.append(kde_H0)
        print()

    return all_a, all_b, all_kde_H1, all_kde_H0

def compute_log_lr_KDE(eval_loader, expert, a, b, kde_H1, kde_H0):
    logits_list, label_list, _, ids, attack_ids = expert_loop_calibration(eval_loader, expert, device, is_test=True)

    logits = np.array(logits_list)
    probs = 1 / (1 + np.exp(-(a * logits + b)))
    probs = np.clip(probs, 1e-6, 1 - 1e-6).reshape(-1, 1)

    log_p_H1 = kde_H1.score_samples(probs)
    log_p_H0 = kde_H0.score_samples(probs)
    x = log_p_H1 - log_p_H0

    return ids, attack_ids, x, label_list, logits

def run_eval_KDE(experts, eval_loader, all_a, all_b, all_H1, all_H0, config):
    all_results = []

    for i, expert in enumerate(experts):
        print(f"EVALUATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)
        eval_loader.dataset.mod = config["all_manipulations"][i]

        a = all_a[i]
        b = all_b[i]
        kde_H1 = all_H1[i]
        kde_H0 = all_H0[i]

        ids, attack_ids, logLR, labels, raw_logits = compute_log_lr_KDE(eval_loader, expert, a, b, kde_H1, kde_H0)
        print(set(labels))

        all_results.append({
            "ids": np.array(ids),
            "attack_ids": np.array(attack_ids),
            "logLR": np.array(logLR),
            "labels": np.array(labels),
            "raw_logits": np.array(raw_logits),
        })
        print()

    print(f"Computing ensemble scores...")

    # Expert 0 (F0-based) defines the valid test subset
    # We remove for all experts the samples where F0 extraction was not successful
    common_ids = all_results[0]["ids"]
    aligned_attack_ids = all_results[0]["attack_ids"]
    labels_ref = all_results[0]["labels"]

    aligned_logLR = []
    aligned_raw_logits = []

    for res in all_results:
        ids = res["ids"]
        logLR = res["logLR"]
        raw_logits = res["raw_logits"]

        # Map IDs to indices for this expert
        id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}

        # Select only samples kept by expert 0
        idxs = [id_to_idx[id_] for id_ in common_ids]

        aligned_logLR.append(logLR[idxs])
        aligned_raw_logits.append(raw_logits[idxs])

    # Stack for downstream use
    all_logLR = np.stack(aligned_logLR, axis=0)  # (5, N)
    all_raw_logits = np.stack(aligned_raw_logits, axis=0)  # (5, N)

    # Save ensemble results
    np.save(f"{config['save_path']}/scores_per_expert.npy", all_logLR)
    np.save(f"{config['save_path']}/raw_logits_per_expert.npy", all_raw_logits)
    np.save(f"{config['save_path']}/labels.npy", np.array(labels_ref))
    np.save(f"{config['save_path']}/attack_ids.npy", aligned_attack_ids)
    np.save(f"{config['save_path']}/filenames.npy", common_ids)

    print("Finished. \n")
