import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.data_utils import load_evaluation_data
from models.small_models import *
from utils.training_utils import eval_expert
from utils.utils import *
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(checkpoint_paths, model_classes):

    # DEFINE MODEL
    experts = []
    for model_path, ModelClass in zip(checkpoint_paths, model_classes):
        expert = ModelClass(return_emb=False)
        expert = expert.to(device)
        expert.load_state_dict(torch.load(model_path, map_location=device))
        experts.append(expert)
        print(f'Model loaded: {os.path.basename(model_path)}\n')

    # LOAD EVAL DATA
    model_names = []
    for i in range(len(checkpoint_paths)):
        name = Path(checkpoint_paths[i]).stem.rsplit('_', 1)[-1]
        model_names.append(name)

    targets = config['selected']
    for target in targets:

        config['selected'] = [target]

        if config['target_check']:
            config['save_path'] = os.path.join(config['save_folder'], target)
        else:
            config['save_path'] = config['save_folder']

        Path(config['save_path']).mkdir(parents=True, exist_ok=True)
        print(f"Result path: {config['save_path']}")

        eval_loader = load_evaluation_data(model_names, config)

        for expert, model_path in zip(experts, checkpoint_paths):
            with torch.no_grad():
                file_IDs, scores, labels = eval_expert(expert, eval_loader, device)

            scores = torch.tensor(scores)
            scores = scores.cpu().numpy()
            labels = np.array(labels)

            print(f"Valid samples: {len(labels)}")

            # fake_scores = scores[labels == 1]
            # real_scores = scores[labels == 0]
            # print(f"Mean score (fake=1): {fake_scores.mean():.6f}")
            # print(f"Mean score (real=0): {real_scores.mean():.6f}")

            rocauc, eer = compute_metrics_v2(labels, scores)
            print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n")

            np.save(f"{config['save_path']}/file_IDs.npy", file_IDs)
            np.save(f"{config['save_path']}/scores.npy", scores)
            np.save(f"{config['save_path']}/labels.npy", labels)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    # --------------------- PLEASE DEFINE ----------------------#

    config['mod'] = 'energy'
    config['sanity_check'] = False
    config['target_check'] = True

    config['train_data'] = 'asv19'
    config['test_data'] = 'asv19'

    config['model'] = 'MLP'
    model_classes = [MLP_Energy]

    model_names = [
        f'{config['mod'].upper()}_{config['model'].upper()}.pth',
    ]

    # ------------------------- Thanks -------------------------#

    if config['target_check']:
        if config['test_data'] == 'asv19':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
        elif config['test_data'] == 'asv5':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A14']
            # config['selected'] = ['A14']
        elif config['test_data'] == 'wf':
            config['selected'] = ["tts_ljspeech", "ljspeech_full_band_melgan", "ljspeech_hifiGAN", "ljspeech_melgan",
                                  "ljspeech_melgan_large", "ljspeech_multi_band_melgan", "ljspeech_parallel_wavegan",
                                  "ljspeech_waveglow", "jsut_multi_band_melgan", "jsut_parallel_wavegan"]
        else:
            print("Test dataset not supported. \n")
            sys.exit()
    else:
        config['selected'] = [config['mod']]

    print("\nEVALUATION MODE\n")
    print(f"Manipulation type: {config['mod']}")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")

    checkpoint_paths = [os.path.join(config['save_model_folder'], 'experts', name) for name in model_names]
    config['save_folder'] = os.path.join(config['save_results_folder'],
                                       'experts',
                                       config['train_data'],
                                       config['mod'])
    if config['target_check']:
        config['save_folder'] = os.path.join(config['save_folder'], 'TargetCheck')
    else:
        config['save_folder'] = os.path.join(config['save_folder'], 'SanityCheck')

    main(checkpoint_paths, model_classes)



