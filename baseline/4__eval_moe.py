import sys
import os
sys.path.insert(1, os.getcwd())

from utils.utils import read_yaml, seed_everything
from pathlib import Path

from utils.data_utils import LoadEvalData_Baseline, load_evaluation_data
from utils.training_utils import eval_moe

from torch.utils.data import DataLoader

import torch
from models.lcnn_model import LCNN
from models.resnet_model import ResNet_LogSpec, ResNet_MelSpec
from models.moe_models import MOE_ICASSP, MOE_WIFS

import warnings
for w in [FutureWarning, UserWarning]:
    warnings.simplefilter("ignore", w)

# sys.path.append("pydevd-pycharm.egg")
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9000, stdout_to_server=True, stderr_to_server=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(checkpoint_path, config):
    Path(config['save_results_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MODEL
    if config['moe_configuration'] == 'mele':
        experts = [LCNN(return_emb=True) for _ in range(4)]
        moe_model = MOE_ICASSP(experts=experts, return_weights=True)

    else:
        experts = [
            LCNN(return_emb=True),
            ResNet_LogSpec(return_emb=True),
            ResNet_MelSpec(return_emb=True),
        ]
        moe_model = MOE_WIFS(experts=experts, return_weights=True)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    # # Block needed for MELE_ORIGINAL.pth
    # state_dict = torch.load(checkpoint_path)
    # new_state_dict = {}
    # mapping = {
    #     'fc1': 'proj.0.0',
    #     'fc2': 'proj.1.0',
    #     'fc3': 'proj.2.0',
    #     'fc4': 'proj.3.0',
    # }
    # for k, v in state_dict.items():
    #     for old_name, new_name in mapping.items():
    #         if k.startswith(old_name):
    #             # replace fcX -> proj.Y.0
    #             new_k = k.replace(old_name, new_name)
    #             new_state_dict[new_k] = v
    #             break
    #     else:
    #         new_state_dict[k] = v
    # moe_model.load_state_dict(new_state_dict, strict=False)

    moe_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('EVALUATION - Model loaded : {}'.format(os.path.basename(checkpoint_path)))

    # LOAD EVAL DATA
    datasets = ['asv19', 'itw']
    eval_sets = load_evaluation_data(datasets, config)

    df_asv19 = eval_sets['asv19']
    df_itw = eval_sets['itw']

    # EVALUATE MODEL
    for dataset, df_eval in zip(datasets, [df_asv19, df_itw]):

        save_dir = os.path.join(config['save_results_folder'], dataset)
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing: {dataset.upper()}\n")

        result_path = os.path.join(save_dir, os.path.basename(checkpoint_path).replace('.pth', '.txt'))

        if os.path.exists(result_path):
            print("Save path exists - Deleting file")
            os.remove(result_path)

        d_label_eval = dict(zip(df_eval['path'], df_eval['label']))
        file_eval = list(df_eval['path'])

        eval_set = LoadEvalData_Baseline(list_IDs=file_eval, labels=d_label_eval, config=config)
        eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False)
        del eval_set, d_label_eval

        with torch.no_grad():
            eval_moe(moe_model, eval_loader, result_path, 'cuda')


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    rootpath = os.path.join(config['rootpath'], 'checkpoints/baseline')

    config['moe_configuration'] = 'mile'
    config['eval_checkpoint'] = 'MILE_ORIGINAL.pth'
    checkpoint_path = os.path.join(rootpath, config['eval_checkpoint'])
    config['save_results_folder'] = os.path.join(config['save_results_folder'], 'baseline')

    main(checkpoint_path, config)



