import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.data_utils import load_training_data
from models.moe_models import *
from utils.training_utils import train_epoch_compound, valid_epoch_compound
from utils.utils import *
from utils.model_utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_moe(ckpts, config):
    config['save_model_folder'] = os.path.join(config['save_model_folder'], 'MoE')
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    # LOAD EXPERT MODELS
    experts = []
    models_list = []
    model_registry = MODEL_REGISTRY.copy()

    for ckpt in ckpts:
        model_type = get_model_type_from_ckpt(ckpt)
        for name, mod in model_registry:
            if name == model_type:
                model_class = mod
                model_registry.remove((name, model_class))  # remove the used pair to keep mapping one-to-one
                break
        model = model_class(return_emb=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        experts.append(model)
        models_list.append(model_type)

    scaling_params_path = os.path.join(config['save_results_folder'], 'Ensemble/asv19/experts_trained_on_asv19/NEW')
    a_params = np.load(os.path.join(scaling_params_path, 'all_a.npy'))
    b_params = np.load(os.path.join(scaling_params_path, 'all_b.npy'))
    a_params = torch.tensor(a_params).to(device)
    b_params = torch.tensor(b_params).to(device)

    # DEFINE MOE MODEL
    moe_model = MOE_Way3(experts=experts,
                         a_params=a_params,
                         b_params=b_params,
                         freezing=config['freeze_experts'],
                         return_weights=True)

    for expert in experts:
        expert.to(device)
    moe_model = (moe_model).to(device)

    # LOAD TRAINING DATA
    train_loader, dev_loader = load_training_data(models_list, config)

    # TRAINING & VALIDATION
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, moe_model.parameters()), lr=float(config['lr_moe']), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=float(config['eta_min']))

    losses = nn.ModuleDict({
        "bce": nn.BCEWithLogitsLoss(),
        "nll": nn.NLLLoss(),
    })
    losses = losses.to(device)

    best_metric = 100
    best_epoch = 0
    early_stopping = 0
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")

            train_losses = train_epoch_compound(train_loader, moe_model, optimizer, losses, config, device)
            with torch.no_grad():
                val_losses, scores, labels = valid_epoch_compound(dev_loader, moe_model, losses, config, device)

            scheduler.step()
            rocauc, eer = compute_metrics_v2(labels, scores)

            print(f'Epoch: {epoch} \n '
                  f'Train Losses - TOT: {train_losses['total']:.5f}, BCE: {train_losses['bce']:.5f}, ATTR: {train_losses['attr']:.5f} \n '
                  f'Val Losses - TOT: {val_losses['total']:.5f}, BCE: {val_losses['bce']:.5f}, ATTR: {val_losses['attr']:.5f} \n '
                  f'EER: {eer * 100:.2f}%, AUC: {rocauc * 100:.2f}%  \n ')

            # if val_losses['total'] < best_metric:
            if eer < best_metric:
                print(f'Best model found at epoch {epoch}')
                torch.save(moe_model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
                early_stopping = 0
                best_epoch = epoch
                # best_metric = min(val_losses['total'], best_metric)
                best_metric = min(eer, best_metric)
            else:
                early_stopping += 1
        else:
            print(f'Training stopped after {epoch} epochs ')
            break

        if early_stopping >= config['early_stopping']:
            print(f"Early stopping triggered. Best validation metric: {best_metric:.6f} at epoch {best_epoch} \n")
        else:
            print(f"Best validation metric: {best_metric:.6f} at epoch {best_epoch} \n")


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])

    def setup_moe_model(model_name, freeze_experts=True):
        config['train_data'] = 'asv19'
        config['merge'] = True
        config['reduce_set'] = False

        config['freeze_experts'] = freeze_experts
        config['model_name'] = model_name
        ckpts_path = os.path.join(config['rootpath'], 'checkpoints/experts')

        ckpts = [
            os.path.join(ckpts_path, 'FLAT_PITCH_MLP.pth'),
            os.path.join(ckpts_path, 'F0_INCONS_MLP.pth'),
            os.path.join(ckpts_path, 'OVERSMOOTH_MLP.pth'),
            os.path.join(ckpts_path, 'MIN_PHASE_LCNN.pth'),
            os.path.join(ckpts_path, 'ENERGY_MLP.pth')
        ]

        print(f"\nModel name: {config['model_name']}")
        print(f"Freeze experts: {freeze_experts}\n")

        train_moe(ckpts, config)


    setup_moe_model(model_name='MoE_Way3_Asv19_Frozen_Scaled.pth',
                    freeze_experts=True)