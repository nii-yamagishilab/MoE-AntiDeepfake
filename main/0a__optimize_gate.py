import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from pathlib import Path

from models.lcnn_model import LCNN
from models.moe_models import MOE_ICASSP, MOE_WIFS, FlatteningLoss
from utils.data_utils import load_asv5_bonafide, LoadExperimentalData
from utils.training_utils import train_epoch_compound, valid_epoch_compound
from utils.utils import *
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_moe(config):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'gate_exp')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    expert_ckpts_path = os.path.join(config['rootpath'], 'checkpoints/POC')

    # DEFINE MOE MODEL
    experts = [LCNN(return_emb=True) for _ in range(4)]
    ckpts = [
        os.path.join(expert_ckpts_path, 'POC_LCNN_0-2000.pth'),
        os.path.join(expert_ckpts_path, 'POC_LCNN_2000-4000.pth'),
        os.path.join(expert_ckpts_path, 'POC_LCNN_4000-6000.pth'),
        os.path.join(expert_ckpts_path, 'POC_LCNN_6000-8000.pth')
    ]
    for expert, ckpt in zip(experts, ckpts):
        expert.load_state_dict(torch.load(ckpt, map_location=device))

    if 'cat' in config['model_name'].lower():
        moe_model = MOE_ICASSP(experts=experts, return_weights=True)
    else:
        moe_model = MOE_WIFS(experts=experts, return_weights=True)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    # LOAD TRAINING DATA
    df_asv5 = load_asv5_bonafide(
        [os.path.join(config['asvspoof_5_path'], 'protocols/ASVspoof5.train.tsv'),
            os.path.join(config['asvspoof_5_path'], 'protocols/ASVspoof5.dev.track_1.tsv')],
        config['asvspoof_5_path'])

    df_train, df_dev = train_test_split(
        df_asv5,
        test_size=0.2,
        random_state=config['seed'],
        shuffle=True
    )

    # Define training dataloader
    file_train = list(df_train['path'])
    train_set = LoadExperimentalData(list_IDs=file_train, config=config, multi_band=True)
    train_loader = DataLoader(train_set, batch_size=config['batch_size_moe'], shuffle=True, drop_last=True)
    del train_set

    # Define validation dataloader
    file_dev = list(df_dev['path'])
    dev_set = LoadExperimentalData(list_IDs=file_dev, config=config, multi_band=True)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size_moe'], shuffle=False)
    del dev_set

    print(f'Total no. of training samples: {len(train_loader.dataset)}')
    print(f'Total no. of dev samples: {len(dev_loader.dataset)} \n')

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

    def setup_moe_model(model_name, moe_configuration='mele', freeze_experts=True):
        config['moe_configuration'] = moe_configuration
        config['freeze_experts'] = freeze_experts
        config['model_name'] = model_name

        print(f"\nMoE configuration: {moe_configuration}")
        print(f"Model name: {config['model_name']}")
        print(f"Freeze experts: {freeze_experts}\n")

        train_moe(config)


    setup_moe_model(model_name='MELE_CAT_COMPOUND.pth',
                    moe_configuration='mele',
                    freeze_experts=False)