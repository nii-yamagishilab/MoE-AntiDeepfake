import sys
import os
sys.path.insert(1, os.getcwd())
from pathlib import Path
from models.moe_models import MOE_ICASSP, MOE_WIFS
from utils.data_utils import load_evaluation_data
from utils.training_utils import eval_moe_compound
from utils.utils import *
from utils.model_utils import MODEL_REGISTRY, get_model_type_from_ckpt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_moe(checkpoint_moe, ckpts_experts, save_folder, config):
    Path(save_folder).mkdir(parents=True, exist_ok=True)

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
        # model.load_state_dict(torch.load(ckpt, map_location=device)) # loaded with the moe later
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

    targets = ['A01', 'A02', 'A03', 'A04', 'A07',  'A08', 'A09', 'A10', 'A11', 'A12']
    for t in range(len(targets)):
        target = targets[t]
        config['selected_target'] = [target]
        result_path = os.path.join(save_folder, f"{config['selected_target'][0]}", os.path.basename(os.path.splitext(checkpoint_moe)[0]))
        Path(result_path).mkdir(parents=True, exist_ok=True)
        print(f"Result path: {result_path}")

    # config['selected_target'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
    # result_path = os.path.join(save_folder, 'ALL', os.path.basename(os.path.splitext(checkpoint_moe)[0]))
    # Path(result_path).mkdir(parents=True, exist_ok=True)
    # print(f"Result path: {result_path}\n")

        # LOAD EVAL DATA
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

    config['sanity_check'] = False
    if config['sanity_check']:
        # config['selected_mods'] = ['f0_incons', 'flat_pitch', 'oversmooth', 'min_phase', 'perf_sil']
        config['selected_mods'] = ['min_phase']
    config['target_check'] = True

    ckpt_moe = os.path.join(ckpt_moe_path, 'frozen_5_exp_wxL_no_temp.pth')
    ckpts_experts = [
        os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP.pth'),
        os.path.join(ckpt_exp_path, 'F0_INCONS_MLP.pth'),
        os.path.join(ckpt_exp_path, 'OVERSMOOTH_GRU.pth'),
        os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN.pth'),
        os.path.join(ckpt_exp_path, 'PERF_SIL_LCNN.pth')
    ]

    # save_folder = os.path.join(save_path, "Full_ASV19")
    # save_folder = os.path.join(save_path, "SanityCheck")
    save_folder = os.path.join(save_path, "TargetCheck")

    # ------------------------- Thanks -------------------------#

    print("\nEVALUATION MODE\n")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")

    test_moe(ckpt_moe, ckpts_experts, save_folder, config)




