import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.data_utils import load_training_data
from models.feat_1D_models import *
from models.lcnn_model import LCNN
from models.moe_models import MOE_ICASSP, MOE_WIFS, FlatteningLoss
from utils.training_utils import train_epoch_compound, valid_epoch_compound
from utils.utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODEL_REGISTRY = {
    "LSTM": LSTM_Model,
    "BILSTM": BiLSTM_Model,
    "GRU": GRU_Model,
    "LCNN": LCNN,
}


def get_model_type_from_ckpt(ckpt_path):
    filename = os.path.basename(ckpt_path).upper()
    name_no_ext = os.path.splitext(filename)[0]

    tokens = name_no_ext.split("_")
    for token in tokens:
        token_upper = token.upper()
        if token_upper in MODEL_REGISTRY:
            return token_upper

    raise ValueError(f"No model recognized from checkpoint: {filename}")


def train_moe(ckpts, config):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'gate_exp')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # LOAD EXPERT MODELS
    experts = []
    models_list = []
    for ckpt in ckpts:
        model_type = get_model_type_from_ckpt(ckpt)
        model_class = MODEL_REGISTRY[model_type]
        model = model_class(return_emb=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        experts.append(model)
        models_list.append(model_class)

    # DEFINE MOE MODEL
    if 'cat' in config['model_name'].lower():
        moe_model = MOE_ICASSP(experts=experts, return_weights=True)
    else:
        moe_model = MOE_WIFS(experts=experts, return_weights=True)

    for expert in experts:
        expert.to(device)
    moe_model = (moe_model).to(device)

    # LOAD TRAINING DATA
    train_loader, dev_loader = load_training_data(models_list, config)

    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, moe_model.parameters()), lr=float(config['lr']), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=0.00001)

    losses = nn.ModuleDict({
        "bce": nn.BCELoss(),
        "nll": nn.NLLLoss(),
        "flatten": FlatteningLoss(),
    })
    losses = losses.to(device)

    best_loss = 100
    best_epoch = 0
    early_stopping = 0
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")

            train_losses = train_epoch_compound(train_loader, moe_model, optimizer, losses, config, device)
            with torch.no_grad():
                val_losses = valid_epoch_compound(dev_loader, moe_model, losses, config, device)

            scheduler.step()

            print(f'Epoch: {epoch} \n '
                  f'Train Losses - TOT: {train_losses['total']:.5f}, BCE: {train_losses['bce']:.5f}, ATTR: {train_losses['attr']:.5f}, FLAT: {train_losses['flat']:.5f} \n '
                  f'Val Losses - TOT: {val_losses['total']:.5f}, BCE: {val_losses['bce']:.5f}, ATTR: {val_losses['attr']:.5f}, FLAT: {val_losses['flat']:.5f} \n ')

            if val_losses['total'] < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(moe_model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
                early_stopping = 0
                best_epoch = epoch
                best_loss = min(val_losses['total'], best_loss)
            else:
                early_stopping += 1
        else:
            print(f'Training stopped after {epoch} epochs ')
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

    def setup_moe_model(model_name, freeze_experts=True):
        config['freeze_experts'] = freeze_experts
        config['model_name'] = model_name
        ckpts_path = os.path.join(config['rootpath'], 'checkpoints/experts')

        ckpts = [
            os.path.join(ckpts_path, 'POC_LCNN_0-2000.pth'),
            os.path.join(ckpts_path, 'POC_LCNN_2000-4000.pth'),
            os.path.join(ckpts_path, 'POC_LCNN_4000-6000.pth'),
            os.path.join(ckpts_path, 'POC_LCNN_6000-8000.pth')
        ]

        print(f"Model name: {config['model_name']}")
        print(f"Freeze experts: {freeze_experts}\n")

        train_moe(ckpts, config)


    setup_moe_model(model_name='MOE_COMPOUND_LOSS.pth',
                    freeze_experts=True)