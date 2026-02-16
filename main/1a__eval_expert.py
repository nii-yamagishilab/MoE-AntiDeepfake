import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from pathlib import Path
from utils.data_utils import load_evaluation_data
from models.feat_1D_models import BiLSTM_Model, LSTM_Model, GRU_Model
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
        print(f'\nEVALUATION - Model loaded: {os.path.basename(model_path)}\n')

    # LOAD EVAL DATA
    eval_loader = load_evaluation_data(config)

    # EVALUATE EXPERT(S)
    for expert, model_path in zip(experts, checkpoint_paths):

        with torch.no_grad():
            scores, labels = eval_expert(expert, eval_loader, device)

        rocauc, eer = compute_metrics(labels, scores)
        print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n")


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    config['mod'] = 'flat_pitch'
    config['model'] = 'gru'
    config['sanity'] = False
    model_classes = [GRU_Model]

    model_names = [
        f'{config['mod'].upper()}_{config['model'].upper()}.pth',
    ]

    # config['save_results_folder'] = os.path.join(config['save_results_folder'], 'experts')
    checkpoint_paths = [os.path.join(config['save_model_folder'], 'experts', name) for name in model_names]

    main(checkpoint_paths, model_classes)



