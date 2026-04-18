import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import os
sys.path.insert(1, os.getcwd())

from utils.data_utils import *
from pathlib import Path
from utils.training_utils import expert_loop_calibration
from utils.utils import *
from utils.model_utils import MODEL_REGISTRY, get_model_type_from_ckpt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cmlg_scaling(calib_loader, expert):
    logits_list, label_list, all_emb_r, all_emb_f = expert_loop_calibration(calib_loader, expert, device)

    # mean_r = all_emb_r.mean(axis=0)  # (D,)
    # mean_f = all_emb_f.mean(axis=0)  # (D,)
    # centroids = np.stack([mean_r, mean_f], axis=0)  # (2, D)

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

def run_calibration(experts, config):
    all_a = []
    all_b = []
    all_mf = []
    all_mr = []
    all_v = []
    # all_centroids = []

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

        # all_centroids.append(centroids)
        print()

    return all_a, all_b, all_mf, all_mr, all_v

def compute_log_lr(eval_loader, expert, a, b):
    logits_list, label_list, all_emb, ids, attack_ids = expert_loop_calibration(eval_loader, expert, device, is_test=True)
    logits = np.array(logits_list)
    x = a * logits + b   # calibrated log-likelihood-ratio
    return ids, attack_ids, x, label_list, logits, all_emb

def run_evaluation(experts, eval_loader, all_a, all_b, config):
    all_results = []

    for i, expert in enumerate(experts):
        print(f"EVALUATING EXPERT {i+1} / {len(experts)}...")
        expert.to(device)
        eval_loader.dataset.mod = config["all_manipulations"][i]

        a = all_a[i]
        b = all_b[i]

        ids, attack_ids, logLR, labels, raw_logits, all_emb = compute_log_lr(eval_loader, expert, a, b)

        all_results.append({
            "ids": np.array(ids),
            "attack_ids": np.array(attack_ids),
            "logLR": np.array(logLR),
            "labels": np.array(labels),
            "raw_logits": np.array(raw_logits),
            "embeddings": np.array(all_emb),
        })
        print()

    print(f"Computing ensemble scores...")

    # Keep only samples valid for all F0-based experts (here first one must be F0-based!)
    common_ids = all_results[0]["ids"]
    aligned_logLR = []
    aligned_raw_logits = []
    aligned_embeddings = []
    aligned_attack_ids = []
    labels_ref = None

    for res in all_results:
        ids = res["ids"]
        attack_ids = res["attack_ids"]
        logLR = res["logLR"]
        labels = res["labels"]
        raw_logits = res["raw_logits"]
        embeddings = res["embeddings"]  # (N_i, D)

        id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        idxs = [id_to_idx[id_] for id_ in common_ids]

        aligned_logLR.append(logLR[idxs])
        aligned_raw_logits.append(raw_logits[idxs])
        aligned_embeddings.append(embeddings[idxs])
        aligned_attack_ids = [attack_ids[idxs]]

        if labels_ref is None:
            labels_ref = labels[idxs]

    all_logLR = np.stack(aligned_logLR)  # shape: (n_experts, N)
    all_raw_logits = np.stack(aligned_raw_logits)

    # Save ensemble results
    np.save(f"{config['save_path']}/scores_per_expert.npy", all_logLR)
    np.save(f"{config['save_path']}/raw_logits_per_expert.npy", all_raw_logits)
    np.save(f"{config['save_path']}/labels.npy", np.array(labels_ref))
    np.save(f"{config['save_path']}/attack_ids.npy", aligned_attack_ids)
    np.save(f"{config['save_path']}/filenames.npy", common_ids)

    print("Finished. \n")

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
        model = model_class(return_emb=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        experts.append(model)
        models_list.append(model_type)

    # _ = load_evaluation_data(models_list, config)  # debug

    # Calibrate & evaluate
    Path(config['save_path']).mkdir(parents=True, exist_ok=True)
    print(f"Result path: {config['save_path']} \n")

    names = ['all_a', 'all_b', 'all_mf', 'all_mr', 'all_v']
    files = {name: f"{config['save_path']}/{name}{'.pkl' if 'H' in name else '.npy'}" for name in names}

    values = dict(zip(names, run_calibration(experts, config)))

    for k in ['all_a', 'all_b', 'all_mf', 'all_mr', 'all_v']:
        np.save(files[k], values[k])

    eval_loader = load_evaluation_data(models_list, config)
    run_evaluation(experts, eval_loader, values['all_a'], values['all_b'], config)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")
    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])

    ckpt_exp_path = os.path.join(config['save_model_folder'], 'experts')
    save_path = os.path.join(config['save_results_folder'], 'Ensemble')

    # --------------------- PLEASE DEFINE ----------------------#

    config['train_data'] = 'mix'
    config['calibration_data'] = 'asv19'
    config['test_data'] = 'mlaad'

    config['pool_calibration_set'] = True

    config["all_manipulations"] = ['f0_incons', 'oversmooth',
                                   'min_phase', 'energy', 'ckrboard']  # mind the order!!!

    config['save_path'] = os.path.join(save_path,
                                         config['test_data'],
                                         f'experts_trained_on_{config["train_data"]}',
                                         '5_experts_FULL_EVAL/'
                                         'Cset_ASV19_pooled')

    # ------------------------- Thanks -------------------------#

    print("\nTEST CALIBRATED ENSEMBLE\n")

    if config['train_data'] == 'asv19':
        ckpts_experts = [
            os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'F0_INCONS_MLP.pth'),
            os.path.join(ckpt_exp_path, 'OVERSMOOTH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN.pth'),
            os.path.join(ckpt_exp_path, 'ENERGY_MLP.pth'),
            os.path.join(ckpt_exp_path, 'CKRBOARD_MLP.pth')
        ]
    else:
        ckpts_experts = [
            # os.path.join(ckpt_exp_path, 'MIX_FLAT_PITCH_MLP.pth'),
            # os.path.join(ckpt_exp_path, 'MIX_F0_INCONS_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIX_F0_JOINT_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIX_OVERSMOOTH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIX_MIN_PHASE_LCNN.pth'),
            os.path.join(ckpt_exp_path, 'MIX_ENERGY_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIX_TONAL_MLP.pth')
        ]

    if config['test_data'] == 'asv19':
        config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
    elif config['test_data'] == 'asv5':
        config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A14', 'A17', 'A19', 'A21', 'A22', 'A28', 'A29']
    elif config['test_data'] == 'sc':
        config['selected'] = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23']
    elif config['test_data'] == 'mlaad':
        config['selected'] = ['Mars5', 'MatchaTTS', 'MeloTTS', 'Metavoice-1B', 'OpenVoiceV2', 'WhisperSpeech', 'e2-tts', 'f5-tts', 'facebook_mms-tts-eng', 'griffin_lim', 'microsoft_speecht5_tts', 'optispeech',
                              'parler_tts_large_v1', 'parler_tts_mini_v0.1', 'parler_tts_mini_v1', 'suno_bark-small', 'suno_bark', 'tts_models_en_blizzard2013_capacitron-t2-c50',
                              'tts_models_en_ek1_tacotron2', 'tts_models_en_jenny_jenny', 'tts_models_en_ljspeech_fast_pitch', 'tts_models_en_ljspeech_glow-tts',
                              'tts_models_en_ljspeech_neural_hmm', 'tts_models_en_ljspeech_overflow', 'tts_models_en_ljspeech_speedy-speech', 'tts_models_en_ljspeech_tacotron2-DCA', 'tts_models_en_ljspeech_tacotron2-DDC',
                              'tts_models_en_ljspeech_tacotron2-DDC_ph', 'tts_models_en_ljspeech_vits--neon', 'tts_models_en_ljspeech_vits', 'tts_models_en_multi-dataset_tortoise-v2',
                              'tts_models_en_sam_tacotron-DDC', 'tts_models_multilingual_multi-dataset_bark', 'tts_models_multilingual_multi-dataset_xtts_v1.1', 'tts_models_multilingual_multi-dataset_xtts_v2', 'vixTTS']
    else:
        print("Test dataset not supported.\n")
        sys.exit()

    config['target_check'] = True
    config['sanity_check'] = False

    if config['train_data'] == 'asv5':
        config['reduce_set'] = True
    else:
        config['reduce_set'] = False

    if ['calibration_data'] == 'pseudo':
        config['pool_calibration_set'] = False

    test_ensemble(ckpts_experts, config)



## WEIGHTING MECHANISM:
#
# # Core function:
# def compute_weights_cosine_per_expert(emb, centroids, tau=0.1, tau_out=0.3, eps=1e-8):
#     # emb: (N, D_i), centroids: (2, D_i)
#     c_r, c_f = centroids
#
#     # normalize
#     emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + eps)
#     c_r = c_r / (np.linalg.norm(c_r) + eps)
#     c_f = c_f / (np.linalg.norm(c_f) + eps)
#
#     # cosine distance
#     d_r = 1 - np.sum(emb * c_r, axis=1)
#     d_f = 1 - np.sum(emb * c_f, axis=1)
#
#     # WEIGHT DECAY
#     d_min = np.minimum(d_r, d_f)
#     w = np.exp(-d_min / tau)  # (N,)
#
#     # print(np.percentile(d_min, [50, 75, 90, 95, 99]))
#
#     # # OPT-OUT
#     # w[d_min > tau_out] = 0.0
#
#     return w
#
# # Put in run_evaluation:
# weights = []
# for i in range(len(experts)):
#     w_i = compute_weights_cosine_per_expert(
#         aligned_embeddings[i],
#         all_centroids[i],
#         tau=0.05,
#         tau_out=0.05
#     ) # (N,)
#     weights.append(w_i)
# weights = np.stack(weights)  # (n_experts, N)
# weights /= (np.sum(weights, axis=0, keepdims=True) + 1e-8)  # normalize per sample
# np.save(f"{config['save_path']}/weights_per_expert.npy", weights)

