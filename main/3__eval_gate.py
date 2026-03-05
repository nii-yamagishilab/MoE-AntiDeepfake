import sys
import os
sys.path.insert(1, os.getcwd())
from pathlib import Path
from models.moe_models import MOE_ICASSP, MOE_WIFS, MOE_WIFS_v2
from utils.data_utils import load_evaluation_data
from utils.training_utils import eval_moe_compound
from utils.utils import *
from utils.model_utils import MODEL_REGISTRY, get_model_type_from_ckpt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_eval(moe_model, models_list, result_path, config):
    Path(result_path).mkdir(parents=True, exist_ok=True)
    print(f"Result path: {result_path}")

    eval_loader = load_evaluation_data(models_list, config)

    with torch.no_grad():
        scores, labels = eval_moe_compound(moe_model, eval_loader, result_path, device)

    scores = np.array(scores)
    labels = np.array(labels)

    print(f"Valid samples: {len(labels)}")
    print(f"Num fake: {(labels == 1).sum()}, Num real: {(labels == 0).sum()}")

    fake_scores = scores[labels == 1]
    real_scores = scores[labels == 0]

    print(f"Mean score (fake=1): {fake_scores.mean():.6f}")
    print(f"Mean score (real=0): {real_scores.mean():.6f}")

    rocauc, eer = compute_metrics(labels, scores)
    print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n\n")


def test_moe(checkpoint_moe, ckpts_experts, save_folder, config):
    checkpoint_name = os.path.basename(os.path.splitext(checkpoint_moe)[0])

    # LOAD EXPERT MODELS
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
        experts.append(model)
        models_list.append(model_type)

    # LOAD GATING NETWORK
    if 'cat' in checkpoint_moe.lower():
        moe_model = MOE_ICASSP(experts=experts, return_weights=False)
    else:
        moe_model = MOE_WIFS(experts=experts, return_weights=False)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    moe_model.load_state_dict(torch.load(checkpoint_moe, map_location=device))
    print('Model loaded : {}'.format(os.path.basename(checkpoint_moe)))

    # RUN EVALUATION
    if config['eval_all']:
        result_path = os.path.join(save_folder, "ALL", checkpoint_name)
        run_eval(moe_model, models_list, result_path, config)
    else:
        targets = config['selected']
        for target in targets:
            config['selected'] = [target]
            result_path = os.path.join(save_folder, target, checkpoint_name)
            run_eval(moe_model, models_list, result_path, config)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")
    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])

    ckpt_exp_path = os.path.join(config['save_model_folder'], 'experts')
    ckpt_moe_path = os.path.join(config['save_model_folder'], 'MoE')
    save_path = os.path.join(config['save_results_folder'], 'MoE')

    # --------------------- PLEASE DEFINE ----------------------#

    config['sanity_check'] = True
    config['target_check'] = False

    config['eval_all'] = False

    ckpt_moe = os.path.join(ckpt_moe_path, 'best_frozen_5_exp_new.pth')
    ckpts_experts = [
        os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP.pth'),
        os.path.join(ckpt_exp_path, 'F0_INCONS_MLP.pth'),
        os.path.join(ckpt_exp_path, 'OVERSMOOTH_GRU.pth'),
        os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN.pth'),
        os.path.join(ckpt_exp_path, 'ENERGY_LCNN.pth')
    ]

    save_folder = os.path.join(save_path, "SanityCheck")
    # save_folder = os.path.join(save_path, "TargetCheck")

    # ------------------------- Thanks -------------------------#

    if config['target_check']:
        config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
    else:
        config['selected'] = ['flat_pitch', 'f0_incons', 'oversmooth', 'min_phase', 'energy']

    print("\nEVALUATION MODE\n")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")

    test_moe(ckpt_moe, ckpts_experts, save_folder, config)




