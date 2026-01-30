import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from pathlib import Path

from models.lcnn_model import LCNN
from models.moe_models import MOE_ICASSP, MOE_WIFS
from utils.data_utils import load_asv5_bonafide, LoadExperimentalData
from utils.training_utils import train_epoch, valid_epoch, train_epoch_moe, valid_epoch_moe, eval_moe
from utils.utils import *
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_expert(config):
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MODEL
    if config['model'] == 'LCNN':
        model = LCNN(return_emb=False)
    else:
        raise ValueError(f'Model {config["model"]} not recognized')

    model = (model).to(device)
    model.apply(init_weights)

    if os.path.exists(os.path.join(config['save_model_folder'], config['model_name'])):
        model.load_state_dict(torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))
        print('Model loaded : {}'.format(config['model_name']))


    # LOAD TRAINING DATA
    df_asv5 = load_asv5_bonafide(
        ['/Database/ASVspoof5/protocols/ASVspoof5.train.tsv',
            '/Database/ASVspoof5/protocols/ASVspoof5.dev.track_1.tsv'],
        '/Database/ASVspoof5')

    df_train, df_dev = train_test_split(
        df_asv5,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Define training dataloader
    file_train = list(df_train['path'])
    train_set = LoadExperimentalData(list_IDs=file_train, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    del train_set

    # Define validation dataloader
    file_dev = list(df_dev['path'])
    dev_set = LoadExperimentalData(list_IDs=file_dev, config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False)
    del dev_set

    print(f'Total no. of training samples: {len(train_loader.dataset)}')
    print(f'Total no. of dev samples: {len(dev_loader.dataset)} \n')


    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=0.00001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    early_stopping = 0
    best_loss = 100
    best_epoch = 0
    best_acc = 0

    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, criterion, device)
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(dev_loader, model, criterion, device)

            scheduler.step()
            print(f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: '
                        f'{valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}')

            if valid_loss < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
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

def train_moe(config):
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MOE MODEL
    experts = [LCNN(return_emb=True) for _ in range(4)]
    ckpts = [
        f'{config['save_model_folder']}/POC_LCNN_0-2000.pth',
        f'{config['save_model_folder']}/POC_LCNN_2000-4000.pth',
        f'{config['save_model_folder']}/POC_LCNN_4000-6000.pth',
        f'{config['save_model_folder']}/POC_LCNN_6000-8000.pth',
    ]
    for expert, ckpt in zip(experts, ckpts):
        expert.load_state_dict(torch.load(ckpt, map_location=device))

    moe_model = MOE_ICASSP(experts=experts, freezing=config['freeze_experts'])
    # moe_model = MOE_WIFS(experts=experts, freezing=config['freeze_experts'])

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    # LOAD TRAINING DATA
    df_asv5 = load_asv5_bonafide(
        ['/Database/ASVspoof5/protocols/ASVspoof5.train.tsv',
            '/Database/ASVspoof5/protocols/ASVspoof5.dev.track_1.tsv'],
        '/Database/ASVspoof5')

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

            running_loss, train_accuracy = train_epoch_moe(train_loader, moe_model, optimizer, criterion, device)
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch_moe(dev_loader, moe_model, criterion, device)

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

def test_moe(checkpoint_path, save_folder, config):
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    experts = [LCNN(return_emb=True) for _ in range(4)]
    # moe_model = MOE_ICASSP(experts=experts, return_weights=True)
    moe_model = MOE_WIFS(experts=experts, return_weights=True)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    moe_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('EVALUATION - Model loaded : {}\n'.format(os.path.basename(checkpoint_path)))

    # LOAD EVAL DATA
    df_eval = load_asv5_bonafide(
        ['/Database/ASVspoof5/protocols/ASVspoof5.eval.track_1.tsv'],
        '/Database/ASVspoof5')

    df_eval = df_eval.sample(n=5000, random_state=config['seed']).reset_index(drop=True)

    print(f'Total no. of test samples: {len(df_eval)}\n')

    bands = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]

    # EVALUATE MODEL
    for selected_band in bands:

        band_as_string = f"{selected_band[0]}_{selected_band[1]}"
        band_dir = os.path.join(save_folder, band_as_string)
        os.makedirs(band_dir, exist_ok=True)

        print(f"Processing: {band_as_string}\n")

        result_path = os.path.join(band_dir, os.path.basename(checkpoint_path).replace('.pth', '.txt'))

        if os.path.exists(result_path):
            print("Save path exists - Deleting file")
            os.remove(result_path)

        file_eval = list(df_eval['path'])

        config['band'] = list(selected_band)

        eval_set = LoadExperimentalData(list_IDs=file_eval, config=config, is_eval=True)
        eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False)
        del eval_set

        with torch.no_grad():
            eval_moe(moe_model, eval_loader, result_path, device)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])
    config['save_model_folder'] = '/home/smg/v-viola/moe_antideepfake/checkpoints/POC'


    def setup_expert_model(model='LCNN', band=(0, 2000)):
        config['model'] = model
        config['band'] = list(band)
        config['model_name'] = f"POC_{model}_{band[0]}-{band[1]}.pth"

        print(f"\nModel type: {model}")
        print(f"Model name: {config['model_name']}")
        print(f"Band: {band}\n")

        train_expert(config)

    def setup_moe_model(model_name, moe_configuration='mele', freeze_experts=True):
        config['moe_configuration'] = moe_configuration
        config['freeze_experts'] = freeze_experts
        config['model_name'] = model_name

        print(f"MoE configuration: {moe_configuration}")
        print(f"Model name: {config['model_name']}")
        print(f"Freeze experts: {freeze_experts}\n")

        train_moe(config)

    def evaluate_moe_model(model_name):
        rootpath = os.path.join(config['rootpath'], 'checkpoints/POC')
        checkpoint_path = os.path.join(rootpath, model_name)
        save_path = os.path.join(config['rootpath'], 'results/POC')

        test_moe(checkpoint_path, save_path, config)


    # ===== Choose which to run =====
    # Uncomment one of the following lines:

    # setup_expert_model(model='LCNN', band=(6000, 8000))
    setup_moe_model(model_name='POC_MELE_CAT.pth', moe_configuration='mele', freeze_experts=False)
    # evaluate_moe_model(model_name='POC_MELE_TRANSF.pth')

