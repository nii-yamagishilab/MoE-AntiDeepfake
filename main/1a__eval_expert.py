import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.data_utils import load_evaluation_data
from models.small_models import BiLSTM_Model, LSTM_Model, GRU_Model, MLP_Model, MLP_Model_v2
from models.lcnn_model import LCNN
from utils.training_utils import eval_expert
from utils.utils import *
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

    targets = ['A01', 'A02', 'A03', 'A04', 'A07',  'A08', 'A09', 'A10', 'A11', 'A12']
    # targets = ['A01']
    for t in range(len(targets)):
        config['selected_target'] = [targets[t]]

        eval_loader = load_evaluation_data(model_names, config)

        # EVALUATE EXPERT(S)
        for expert, model_path in zip(experts, checkpoint_paths):

            with torch.no_grad():
                scores, labels = eval_expert(expert, eval_loader, device)

            # scores = torch.tensor(scores)
            # scores = torch.sigmoid(scores)  # logits → probabilities
            # scores = scores.cpu().numpy()
            #
            # labels = np.array(labels)
            #
            # print(f"Valid samples: {len(labels)}")
            # print(f"Num fake: {(labels == 1).sum()}, Num real: {(labels == 0).sum()}")
            #
            # fake_scores = scores[labels == 1]
            # real_scores = scores[labels == 0]
            #
            # print(f"Mean score (fake=1): {fake_scores.mean():.6f}")
            # print(f"Mean score (real=0): {real_scores.mean():.6f}")

            rocauc, eer = compute_metrics(labels, scores)
            print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n")


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

    config['model'] = 'LCNN'
    model_classes = [LCNN]

    # ------------------------- Thanks -------------------------#

    print("\nEVALUATION MODE\n")
    print(f"Manipulation type: {config['mod']}")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")


    model_names = [
        f'{config['mod'].upper()}_{config['model'].upper()}.pth',
    ]
    checkpoint_paths = [os.path.join(config['save_model_folder'], 'experts', name) for name in model_names]

    main(checkpoint_paths, model_classes)



