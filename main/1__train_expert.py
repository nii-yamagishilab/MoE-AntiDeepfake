import sys
import os
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.data_utils import load_asv5_bonafide
from models.lcnn_model import LCNN
from utils.training_utils import train_epoch_expert, valid_epoch_expert
from utils.utils import *
from utils.make_artifacts import LoadTrainData_Pitch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'experts')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    model = LCNN(return_emb=False)
    model = (model).to(device)
    model.apply(init_weights)

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

    # config['batch_size'] = 100
    # file_train = list(df_train['path'])
    # train_set = LoadTrainData_Pitch_DEBUG(list_IDs=file_train, config=config)
    # train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    # del train_set

    # Define training dataloader
    file_train = list(df_train['path'])
    train_set = LoadTrainData_Pitch(list_IDs=file_train, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    del train_set

    # Define validation dataloader
    file_dev = list(df_dev['path'])
    dev_set = LoadTrainData_Pitch(list_IDs=file_dev, config=config)
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

            running_loss, train_accuracy = train_epoch_expert(train_loader, model, optimizer, criterion, 'cuda')
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch_expert(dev_loader, model, criterion, 'cuda')

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

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    config['model'] = 'LCNN'
    config['mod'] = 'pitch'

    config["model_name"] = f'{config['mod'].upper()}_{config["model"]}.pth'

    print(f"\nModel type: {config['model']}")
    print(f"Model name: {config['model_name']}")

    main(config)
