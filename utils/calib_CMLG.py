import numpy as np
import torch
from utils.data_utils import load_calibration_data
from utils.training_utils import expert_loop_calibration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def cmlg_scaling(calib_loader, expert):
    logits_list, label_list, all_emb_r, all_emb_f = expert_loop_calibration(calib_loader, expert, device)

    s = np.array(logits_list)
    y = np.array(label_list)

    s_f = s[y == 1]
    s_r = s[y == 0]

    m_f = s_f.mean()
    m_r = s_r.mean()

    v = 0.5 * np.mean((s_f - m_f) ** 2) + 0.5 * np.mean((s_r - m_r) ** 2)

    a = (m_f - m_r) / v
    b = -a * (m_f + m_r) / 2

    return a, b, m_f, m_r, v

def run_calibration_CMLG(experts, config):
    all_a = []
    all_b = []
    all_mf = []
    all_mr = []
    all_v = []

    if config['pool_calibration_set']:
        calib_loader = load_calibration_data(config['mod'], config)

    for i, expert in enumerate(experts):
        print(f"CALIBRATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)

        if not config['pool_calibration_set']:
            calib_loader = load_calibration_data(config["all_manipulations"][i], config)

        calib_loader.dataset.mod = config["all_manipulations"][i]
        a, b, m_f, m_r, v = cmlg_scaling(calib_loader, expert)

        all_a.append(a)
        all_b.append(b)
        all_mf.append(m_f)
        all_mr.append(m_r)
        all_v.append(v)

        print()

    return all_a, all_b, all_mf, all_mr, all_v

def compute_log_lr_CMLG(eval_loader, expert, a, b):
    logits_list, label_list, _, ids, attack_ids = expert_loop_calibration(eval_loader, expert, device, is_test=True)
    logits = np.array(logits_list)
    x = a * logits + b   # calibrated log-likelihood-ratio
    return ids, attack_ids, x, label_list, logits

def run_eval_CMLG(experts, eval_loader, all_a, all_b, config):
    all_results = []

    for i, expert in enumerate(experts):
        print(f"EVALUATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)
        eval_loader.dataset.mod = config["all_manipulations"][i]

        a = all_a[i]
        b = all_b[i]

        ids, attack_ids, logLR, labels, raw_logits = compute_log_lr_CMLG(eval_loader, expert, a, b)
        print(logLR.shape)
        print(len(labels))

        all_results.append({
            "ids": np.array(ids),
            "attack_ids": np.array(attack_ids),
            "logLR": np.array(logLR),
            "labels": np.array(labels),
            "raw_logits": np.array(raw_logits),
        })
        print()

    print("Computing ensemble scores...")

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

    # Save results
    np.save(f"{config['save_path']}/scores_per_expert.npy", all_logLR)
    np.save(f"{config['save_path']}/raw_logits_per_expert.npy", all_raw_logits)
    np.save(f"{config['save_path']}/labels.npy", np.array(labels_ref))
    np.save(f"{config['save_path']}/attack_ids.npy", aligned_attack_ids)
    np.save(f"{config['save_path']}/filenames.npy", common_ids)

    print(len(labels_ref))
    print(all_logLR.shape)

    print("Finished. \n")