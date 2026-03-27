import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
sys.path.insert(1, os.getcwd())
from pathlib import Path
from utils.utils import *
from utils.data_utils import load_training_data
from models.small_models import *
from models.lcnn_model import LCNN
from utils.training_utils import train_epoch_expert, valid_epoch_expert
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def select_model(model_name):
    if model_name == 'mlp' and config['mod'] == 'flat_pitch':
        model = MLP_Flat_Pitch(return_emb=True)
    elif model_name == 'mlp' and config['mod'] == 'f0_incons':
        model = MLP_F0_Incons(return_emb=True)
    elif model_name == 'mlp' and config['mod'] == 'energy':
        model = MLP_Energy(return_emb=True)
    elif model_name == 'mlp' and config['mod'] == 'oversmooth':
        model = MLP_Smooth(return_emb=True)
    elif model_name == 'cnn':
        model = CNN_1D_Model(return_emb=True)
    elif model_name == 'gru':
        model = GRU_Model(return_emb=True)
    elif model_name == 'lcnn':
        model = LCNN(return_emb=True)
    else:
        print('Model name not recognized.')
        sys.exit()

    model = (model).to(device)
    return model

class Supervised_ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        features = F.normalize(features, dim=1)

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature
        # remove self-comparisons
        mask = torch.eye(B, dtype=torch.bool).to(device)
        sim.masked_fill_(mask, -1e9)
        # create positive mask
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask.masked_fill(mask, 0)
        # log-softmax
        log_prob = F.log_softmax(sim, dim=1)
        # compute loss
        loss = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        return loss.mean()

def main(config):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'experts')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # LOAD MODEL
    model = select_model(config['model'])

    # LOAD TRAINING DATA
    train_loader, dev_loader = load_training_data([config['model'].upper()], config)

    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    criterion = nn.BCEWithLogitsLoss().to(device)
    # supconloss = Supervised_ContrastiveLoss()

    early_stopping = 0
    best_loss = 100
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            running_loss = train_epoch_expert(train_loader, model, optimizer, criterion, device)
            with torch.no_grad():
                valid_loss = valid_epoch_expert(dev_loader, model, criterion, device)

            scheduler.step(valid_loss)
            print(f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: {valid_loss:.5f}')
            if valid_loss < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
                early_stopping = 0
                best_epoch = epoch
                best_loss = min(valid_loss, best_loss)
            else:
                early_stopping += 1
        else:
            print(f'Training stopped after {epoch} epochs - Best Val Loss {best_loss:.2f}')
            break

    if early_stopping >= config['early_stopping']:
        print(f"Early stopping triggered. Best validation loss: {best_loss:.6f} at epoch {best_epoch} \n")
    else:
        print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch} \n")


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])


    # ------------------- PLEASE DEFINE SETTINGS: ------------------ #

    config['train_data'] = 'asv19'
    config['merge'] = True

    config['model'] = 'mlp'
    config['mod'] = 'flat_pitch'

    # ----------------------------- Thanks --------------------------- #

    if config['train_data'] == 'asv5':
        config['reduce_set'] = True
    else:
        config['reduce_set'] = False

    config["model_name"] = f'{config['mod'].upper()}_{config["model"].upper()}_v2.pth'

    print(f"\nModel type: {config['model']}")
    print(f"Model name: {config['model_name']}")

    main(config)


# filename = "/home/smg/v-viola/moe_antideepfake/train_flat_pitch_mlp.out.txt"
# warning_counts = {}
#
# with open(filename, "r") as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith("Warning:"):
#             warning_counts[line] = warning_counts.get(line, 0) + 1
#
# for warning, count in warning_counts.items():
#     print(f"{warning} -> {count} times")
