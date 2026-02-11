import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from pathlib import Path
from models.lcnn_model import LCNN
from models.resnet_model import ResNet_MelSpec, ResNet_LogSpec
from utils.training_utils import eval_expert
from utils.utils import *
import torch
from utils.data_utils import load_evaluation_data, LoadEvalData_Baseline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(checkpoint_paths, model_classes):
    Path(config['save_results_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MODEL
    experts = []
    for model_path, ModelClass in zip(checkpoint_paths, model_classes):
        expert = ModelClass(return_emb=False)
        expert = expert.to(device)
        expert.load_state_dict(torch.load(model_path, map_location=device))
        experts.append(expert)
        print(f'\nEVALUATION - Model loaded: {os.path.basename(model_path)}\n')


    # LOAD EVAL DATA
    datasets = ['asv19', 'itw']
    eval_sets = load_evaluation_data(datasets, config)

    df_asv19 = eval_sets['asv19']
    df_itw = eval_sets['itw']


    # EVALUATE EXPERT(S)
    for expert, model_path in zip(experts, checkpoint_paths):
        for dataset, df_eval in zip(datasets, [df_asv19, df_itw]):

            save_dir = os.path.join(config['save_results_folder'], dataset)
            os.makedirs(save_dir, exist_ok=True)

            print(f"Processing: {dataset.upper()}\n")

            result_path = os.path.join(save_dir, os.path.basename(model_path).replace('.pth', '.txt'))

            if os.path.exists(result_path):
                print("Save path exists - Deleting file")
                os.remove(result_path)

            d_label_eval = dict(zip(df_eval['path'], df_eval['label']))
            file_eval = list(df_eval['path'])

            eval_set = LoadEvalData_Baseline(list_IDs=file_eval, labels=d_label_eval, config=config)
            eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False)
            del eval_set, d_label_eval

            with torch.no_grad():
                eval_expert(expert, eval_loader, result_path, device)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    model_names = [
        'LCNN_ASV19.pth',
    ]
    model_classes = [LCNN]

    config['save_results_folder'] = os.path.join(config['save_results_folder'], 'baseline')
    checkpoint_paths = [os.path.join(config['save_model_folder'], name) for name in model_names]

    main(checkpoint_paths, model_classes)



