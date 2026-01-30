import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from utils.data_utils import *
from utils.training_utils import train_epoch, valid_epoch
from utils.utils import *

from models.lcnn_model import *
from models.resnet_model import *
from models.moe_models import MOE_ICASSP, MOE_WIFS

import torch
import torch.nn as nn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sys.path.append("pydevd-pycharm.egg")
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9000, stdout_to_server=True, stderr_to_server=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config, freeze_experts=True):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'baseline')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MOE MODEL
    if config['moe_configuration'] == 'mele':
        experts = [LCNN(return_emb=True) for _ in range(4)]
        ckpts = [
            f'{config['save_model_folder']}/LCNN_ASV19.pth',
            f'{config['save_model_folder']}/LCNN_FOR.pth',
            f'{config['save_model_folder']}/LCNN_ITW.pth',
        ]
        for expert, ckpt in zip(experts, ckpts):
            expert.load_state_dict(torch.load(ckpt, map_location=device))

        moe_model = MOE_ICASSP(experts=experts, freezing=freeze_experts)

    else:
        experts = [
            LCNN(return_emb=True),
            ResNet_LogSpec(return_emb=True),
            ResNet_MelSpec(return_emb=True),
        ]
        ckpts = [
            f'{config['save_model_folder']}/LCNN.pth',
            f'{config['save_model_folder']}/RESNET_LOGSPEC.pth',
            f'{config['save_model_folder']}/RESNET_MELSPEC.pth',
        ]
        for expert, ckpt in zip(experts, ckpts):
            expert.load_state_dict(torch.load(ckpt, map_location=device))

        moe_model = MOE_WIFS(experts=experts, freezing=freeze_experts)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    if os.path.exists(os.path.join(config['save_model_folder'], config['model_name'])):
        moe_model.load_state_dict(torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))
        print(f'Checkpoint loaded: {config["model_name"]}')


    # LOAD TRAINING DATA
    data_list = ['asv19', 'for', 'itw']
    train_dfs = []
    dev_dfs = []

    for dataset in data_list:
        df_train, df_dev = load_training_data_holdout(dataset, config)
        train_dfs.append(df_train)
        dev_dfs.append(df_dev)

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_dev = pd.concat(dev_dfs, ignore_index=True)

    # Define training dataloader
    label_trn = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])

    train_set = LoadTrainData(list_IDs=file_train, labels=label_trn, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size_moe'], shuffle=True, drop_last=True)
    del train_set, label_trn

    # Define validation dataloader
    label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])

    dev_set = LoadTrainData(list_IDs=file_dev, labels=label_dev, config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size_moe'], shuffle=False)
    del dev_set, label_dev


    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, moe_model.parameters()), lr=float(config['lr']), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=0.00001)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)

    best_acc = 0
    best_loss = 100
    best_epoch = 0
    early_stopping = 0
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")

            running_loss, train_accuracy = train_epoch(train_loader, moe_model, optimizer, criterion, 'cuda')
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(dev_loader, moe_model, criterion, 'cuda')

            scheduler.step()
            print(f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: '
                  f'{valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}')

            if valid_loss < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(moe_model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
                early_stopping = 0
                best_epoch = epoch
                best_loss = min(valid_loss, best_loss)
                best_acc = valid_accuracy
            else:
                early_stopping += 1
        else:
            print(f'Training stopped after {epoch} epochs - Best Val Acc {best_acc:.2f}')
            break

        if early_stopping >= config['early_stopping']:
            print(f"Early stopping triggered. Best validation loss: {best_loss:.6f} at epoch {best_epoch} \n")
        else:
            print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch} \n")


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    config['moe_configuration'] = 'mele'
    freeze_experts = False

    config['model_name'] = f'MOE_TRIAL.pth'
    print(f"Model name: {config['model_name']}")

    main(config, freeze_experts)

