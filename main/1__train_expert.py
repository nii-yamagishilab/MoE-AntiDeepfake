import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.utils import *
from utils.data_utils import load_training_data
from models.small_models import LSTM_Model, BiLSTM_Model, GRU_Model, MLP_Model
from models.lcnn_model import LCNN
from utils.training_utils import train_epoch_expert, valid_epoch_expert

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def select_model(model_name):

    if model_name == 'mlp':
        model = MLP_Model(return_emb=False)
    elif model_name == 'lstm':
        model = LSTM_Model(return_emb=False)
    elif model_name == 'bilstm':
        model = BiLSTM_Model(return_emb=False)
    elif model_name == 'gru':
        model = GRU_Model(return_emb=False)
    elif model_name == 'lcnn':
        model = LCNN(return_emb=False)
    else:
        print('Model name not recognized.')
        sys.exit()

    model = (model).to(device)

    return model

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

    config['model'] = 'lcnn'
    config['mod'] = 'energy'

    # ----------------------------- Thanks --------------------------- #


    config["model_name"] = f'{config['mod'].upper()}_{config["model"].upper()}.pth'

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
