from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from models.aasist_model import AASIST
from models.lcnn_model import LCNN
from models.resnet_model import ResNet_LogSpec, ResNet_MelSpec
from models.rawnet_model import RawNet
from utils.training_utils import train_epoch, valid_epoch
from utils.data_utils import load_training_data_holdout, LoadTrainData
from utils.utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config, training_gata):
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # DEFINE MOE MODEL
    if config['model'] == 'LCNN':
        model = LCNN(return_emb=False)
    elif config['model'] == 'RESNET_LOGSPEC':
        model = ResNet_LogSpec(return_emb=False)
    elif config['model'] == 'RESNET_MELSPEC':
        model = ResNet_MelSpec(return_emb=False)
    elif config['model'] == 'RAWNET':
        model = RawNet(return_emb=False)
    elif config['model'] == 'AASIST':
        model = AASIST(return_emb=False)
    else:
        raise ValueError(f'Model {config["model"]} not recognized')

    model = (model).to(device)
    model.apply(init_weights)

    if os.path.exists(os.path.join(config['save_model_folder'], config['model_name'])):
        model.load_state_dict(torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))
        print('Model loaded : {}'.format(config['model_name']))


    # LOAD TRAINING DATA
    train_dfs = []
    dev_dfs = []
    for dataset in training_gata:
        df_train, df_dev = load_training_data_holdout(dataset, config)
        train_dfs.append(df_train)
        dev_dfs.append(df_dev)

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_dev = pd.concat(dev_dfs, ignore_index=True)

    # Define training dataloader
    label_trn = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])
    train_set = LoadTrainData(list_IDs=file_train, labels=label_trn, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    del train_set, label_trn

    # Define validation dataloader
    label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])
    dev_set = LoadTrainData(list_IDs=file_dev, labels=label_dev, config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False)
    del dev_set, label_dev


    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=0.00001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    early_stopping = 0
    best_loss = 100
    best_epoch = 0
    best_acc = 0
    # writer = set_tensorboard_writer(config['tensorboard_logdir'], run_name=None)
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, criterion, 'cuda')
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(dev_loader, model, criterion, 'cuda')

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


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    config['model'] = 'LCNN'
    training_data = ['itw']

    if isinstance(training_data, (list, tuple)):
        td = "_".join(training_data).upper()
    else:
        td = training_data.upper()

    config["model_name"] = f'{config["model"]}_{td}.pth'

    print(f"\nModel type: {config['model']}")
    print(f"Model name: {config['model_name']}")
    print(f"Training data: {training_data}\n")

    main(config, training_data)
