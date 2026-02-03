import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F


def train_epoch_expert(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

def valid_epoch_expert(data_loader, model, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

    valid_accuracy = (num_correct / num_total) * 100
    running_loss /= num_total
    return running_loss, valid_accuracy

def eval_expert(model, data_loader, save_path, device):
    model.eval()

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
        fname_list = []
        pred_list = []
        label_list = []

        batch_x = batch_x.to(device)

        batch_out = model(batch_x)
        batch_out = nn.Softmax(dim=1)(batch_out)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        fname_list.extend(utt_id)
        pred_list.extend(batch_score.tolist())
        label_list.extend(batch_y.tolist())

        with open(save_path, 'w') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{} {} {}\n'.format(f, pred, lab))
        fh.close()

    print('Scores saved to {}\n'.format(save_path))


def train_epoch_moe(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).float().to(device)
        batch_out = model(batch_x)
        batch_out = batch_out[:, 1]

        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)

        batch_pred = (batch_out >= 0.5).long()
        num_correct += (batch_pred == batch_y.long()).sum(dim=0).item()

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

def valid_epoch_moe(data_loader, model, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).float().to(device)

        batch_out = model(batch_x)
        batch_out = batch_out[:, 1]

        batch_loss = criterion(batch_out, batch_y)
        batch_pred = (batch_out >= 0.5).long()
        num_correct += (batch_pred == batch_y.long()).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

    valid_accuracy = (num_correct / num_total) * 100
    running_loss /= num_total
    return running_loss, valid_accuracy

def eval_moe(model, data_loader, save_path, device):
    model.eval()
    total_gating_weights = None
    batch_count = 0

    npy_path = save_path.replace('.txt', '.npy')
    if os.path.exists(npy_path):
        os.remove(npy_path)
        print(f'[INFO] Removed existing .npy file: {npy_path}')

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)

        batch_out, gating_weights = model(batch_x)
        batch_score = batch_out[:, 1].detach().cpu().numpy()

        if total_gating_weights is None:
            total_gating_weights = np.sum(gating_weights.detach().cpu().numpy(), axis=0)
        else:
            total_gating_weights += np.sum(gating_weights.detach().cpu().numpy(), axis=0)
        batch_count += batch_x.size(0)

        averaged_gating_weights = total_gating_weights / batch_count  # shape: (num_experts)
        np.save(npy_path, averaged_gating_weights)

        with open(save_path, 'a') as fh:
            for f, pred, lab in zip(utt_id, batch_score.tolist(), batch_y.tolist()):
                fh.write('{} {} {}\n'.format(f, pred, lab))

    print('Scores saved to {}\n'.format(save_path))


def train_epoch_compound(train_loader, model, optim, losses, config, device):
    num_tot = 0.0
    num_tot_fakes = 0.0
    num_tot_real = 0.0

    running_bce_loss = 0.0
    running_attr_loss = 0.0
    running_flat_loss = 0.0
    running_tot_loss = 0.0

    model.train()

    for batch_x, batch_y, batch_k in tqdm(train_loader, total=len(train_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).float().to(device)
        batch_k = batch_k.view(-1).long().to(device)

        batch_size = batch_x.size(0)
        fake_mask = (batch_y == 1)
        real_mask = (batch_y == 0)

        num_tot += batch_size
        num_tot_fakes += fake_mask.sum().item()
        num_tot_real += real_mask.sum().item()

        batch_probs, gate_probs = model(batch_x)
        batch_probs = batch_probs[:, 1]
        batch_probs = batch_probs.clamp(1e-8, 1 - 1e-8)
        gate_probs = gate_probs.clamp(1e-8, 1 - 1e-8)

        # BCELoss (Real vs Fake)
        batch_bce_loss = losses['bce'](batch_probs, batch_y)

        # NLLLoss (Attribution Loss, fakes-only)
        log_gate_probs = torch.log(gate_probs + 1e-8)
        if fake_mask.any():
            batch_attr_loss = losses['nll'](log_gate_probs[fake_mask],
                                            batch_k[fake_mask].unsqueeze(1))
        else:
            batch_attr_loss = torch.tensor(0.0, device=batch_x.device)

        # Custom Loss (Flattening Loss, real-only)
        if real_mask.any():
            batch_flat_loss = losses['flatten'](gate_probs[real_mask])
        else:
            batch_flat_loss = torch.tensor(0.0, device=batch_x.device)


        running_bce_loss += batch_bce_loss.item() * batch_size
        running_attr_loss += batch_attr_loss.item() * fake_mask.sum().item()
        running_flat_loss += batch_flat_loss.item() * real_mask.sum().item()

        batch_total_loss = (batch_bce_loss +
                            config['wa']*batch_attr_loss +
                            config['wf']*batch_flat_loss)

        running_tot_loss += batch_total_loss.item() * batch_size

        optim.zero_grad()
        batch_total_loss.backward()
        optim.step()

    running_bce_loss /= num_tot
    running_attr_loss /= max(num_tot_fakes, 1)
    running_flat_loss /= max(num_tot_real, 1)
    running_tot_loss /= num_tot

    epoch_losses = {
        "total": running_tot_loss,
        "bce": running_bce_loss,
        "attr": running_attr_loss,
        "flat": running_flat_loss,
    }
    return epoch_losses

def valid_epoch_compound(data_loader, model, losses, config, device):
    num_tot = 0.0
    num_tot_fakes = 0.0
    num_tot_real = 0.0

    running_bce_loss = 0.0
    running_attr_loss = 0.0
    running_flat_loss = 0.0
    running_tot_loss = 0.0

    model.eval()

    for batch_x, batch_y, batch_k in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).float().to(device)
        batch_k = batch_k.view(-1).long().to(device)

        batch_size = batch_x.size(0)
        fake_mask = (batch_y == 1)
        real_mask = (batch_y == 0)

        num_tot += batch_size
        num_tot_fakes += fake_mask.sum().item()
        num_tot_real += real_mask.sum().item()

        batch_probs, gate_probs = model(batch_x)
        batch_probs = batch_probs[:, 1]
        batch_probs = batch_probs.clamp(1e-8, 1 - 1e-8)
        gate_probs = gate_probs.clamp(1e-8, 1 - 1e-8)

        # BCELoss (Real vs Fake)
        batch_bce_loss = losses['bce'](batch_probs, batch_y)

        # NLLLoss (Attribution Loss, fakes-only)
        log_gate_probs = torch.log(gate_probs + 1e-8)
        if fake_mask.any():
            batch_attr_loss = losses['nll'](log_gate_probs[fake_mask],
                                            batch_k[fake_mask].unsqueeze(1))
        else:
            batch_attr_loss = torch.tensor(0.0, device=batch_x.device)

        # Custom Loss (Flattening Loss, real-only)
        if real_mask.any():
            batch_flat_loss = losses['flatten'](gate_probs[real_mask])
        else:
            batch_flat_loss = torch.tensor(0.0, device=batch_x.device)

        running_bce_loss += batch_bce_loss.item() * batch_size
        running_attr_loss += batch_attr_loss.item() * fake_mask.sum().item()
        running_flat_loss += batch_flat_loss.item() * real_mask.sum().item()

        batch_total_loss = (batch_bce_loss +
                            config['wa'] * batch_attr_loss +
                            config['wf'] * batch_flat_loss)

        running_tot_loss += batch_total_loss.item() * batch_size

    running_bce_loss /= num_tot
    running_attr_loss /= max(num_tot_fakes, 1)
    running_flat_loss /= max(num_tot_real, 1)
    running_tot_loss /= num_tot

    epoch_losses = {
        "total": running_tot_loss,
        "bce": running_bce_loss,
        "attr": running_attr_loss,
        "flat": running_flat_loss,
    }
    return epoch_losses


