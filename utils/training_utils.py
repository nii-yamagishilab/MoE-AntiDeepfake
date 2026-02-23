import torch
from tqdm import tqdm
import numpy as np
import os


def train_epoch_expert(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)

        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss

def valid_epoch_expert(data_loader, model, criterion, device):
    running_loss = 0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)

    running_loss /= num_total
    return running_loss

def eval_expert(model, data_loader, device):
    model.eval()
    fname_list = []
    score_list = []
    label_list = []

    total_removed = 0
    total_seen = 0

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = torch.stack([x.to(device) for x in batch_x])

        batch_out = model(batch_x)
        # batch_out = nn.Softmax(dim=1)(batch_out)
        batch_score = batch_out.cpu().numpy().ravel()
        batch_y = np.array(batch_y)

        valid_mask = batch_y != -1
        total_removed += np.sum(~valid_mask)
        total_seen += len(batch_y)
        if not np.any(valid_mask):
            continue

        fname_list.extend(np.array(utt_id)[valid_mask].tolist())
        score_list.extend(batch_score[valid_mask].tolist())
        label_list.extend(batch_y[valid_mask].tolist())

    print(f"\nRemoved {total_removed} corrupted samples "
          f"out of {total_seen} total "
          f"({100 * total_removed / total_seen:.2f}%).")

    return score_list, label_list


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
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = [x.to(device) for x in batch_x]
        batch_y = batch_y.view(-1).float().to(device)
        batch_k = batch_k.view(-1).long().to(device)

        batch_size = batch_y.size(0)
        fake_mask = (batch_y == 1)
        real_mask = (batch_y == 0)

        num_tot += batch_size
        num_tot_fakes += fake_mask.sum().item()
        num_tot_real += real_mask.sum().item()

        batch_probs, gate_probs = model(batch_x)
        # batch_probs = batch_probs[:, 1]
        batch_probs = batch_probs.clamp(1e-8, 1 - 1e-8)
        gate_probs = gate_probs.clamp(1e-8, 1 - 1e-8)

        # BCELoss (Real vs Fake)
        batch_bce_loss = losses['bce'](batch_probs, batch_y.unsqueeze(-1))

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
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = [x.to(device) for x in batch_x]
        batch_y = batch_y.view(-1).float().to(device)
        batch_k = batch_k.view(-1).long().to(device)

        batch_size = batch_y.size(0)
        fake_mask = (batch_y == 1)
        real_mask = (batch_y == 0)

        num_tot += batch_size
        num_tot_fakes += fake_mask.sum().item()
        num_tot_real += real_mask.sum().item()

        batch_probs, gate_probs = model(batch_x)
        # batch_probs = batch_probs[:, 1]
        batch_probs = batch_probs.clamp(1e-8, 1 - 1e-8)
        gate_probs = gate_probs.clamp(1e-8, 1 - 1e-8)

        # BCELoss (Real vs Fake)
        batch_bce_loss = losses['bce'](batch_probs, batch_y.unsqueeze(-1))

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


def eval_moe_compound(model, data_loader, save_path, device):
    model.eval()

    npy_path = save_path.replace('.txt', '_gating_mean_global.npy')
    npy_fake_path = save_path.replace('.txt', '_gating_mean_fake.npy')
    npy_real_path = save_path.replace('.txt', '_gating_mean_real.npy')
    npy_all_path = save_path.replace('.txt', '_gating_all.npy')

    for p in [npy_path, npy_fake_path, npy_real_path, npy_all_path]:
        if os.path.exists(p):
            os.remove(p)
            print(f'[INFO] Removed existing file: {p}')
    print()

    scores = []
    labels = []
    identifiers = []
    all_weights = []

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            batch_x = values[0] if len(values) == 1 else values

        batch_x = [x.to(device) for x in batch_x]

        batch_probs, gating_probs = model(batch_x)

        batch_probs = batch_probs.detach().cpu().numpy()
        gating_probs = gating_probs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()

        # ---- FILTER INVALID ----
        valid_mask = batch_y != -1
        if np.sum(valid_mask) == 0:
            continue

        batch_probs = batch_probs[valid_mask]
        gating_probs = gating_probs[valid_mask]
        batch_y = batch_y[valid_mask]
        batch_identifiers = [utt_id[i] for i in range(len(utt_id)) if valid_mask[i]]

        all_weights.extend(gating_probs)
        scores.extend(batch_probs)
        labels.extend(batch_y)
        identifiers.extend(batch_identifiers)

    scores = np.array(scores)
    labels = np.array(labels)
    all_weights = np.array(all_weights)

    # ---- GLOBAL MEAN GATING ----
    mean_global = all_weights.mean(axis=0)
    np.save(npy_path, mean_global)

    # ---- CLASS-CONDITIONAL GATING ----
    if (labels == 1).sum() > 0:
        mean_fake = all_weights[labels == 1].mean(axis=0)
        np.save(npy_fake_path, mean_fake)
    else:
        print("No fake samples found.")

    if (labels == 0).sum() > 0:
        mean_real = all_weights[labels == 0].mean(axis=0)
        np.save(npy_real_path, mean_real)
    else:
        print("No real samples found.")

    # ---- SAVE ALL WEIGHTS FOR FUTURE ANALYSIS ----
    np.save(npy_all_path, all_weights)

    # ---- SAVE SCORES FILE ----
    with open(save_path, 'w') as fh:
        for f, pred, lab in zip(identifiers, scores, labels):
            fh.write('{} {} {}\n'.format(f, pred, lab))

    print('\nScores saved to {}'.format(save_path))
    print("Saved gating statistics (.npy files).\n")

    return scores, labels


# def eval_moe_compound(model, data_loader, save_path, device):
#     model.eval()
#     total_gating_weights = None
#     batch_count = 0
#
#     npy_path = save_path.replace('.txt', '.npy')
#     if os.path.exists(npy_path):
#         os.remove(npy_path)
#         print(f'[INFO] Removed existing .npy file: {npy_path}')
#
#     scores = []
#     labels = []
#     identifiers = []
#     for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
#         if isinstance(batch_x, dict):
#             values = list(batch_x.values())
#             if len(values) == 1:
#                 batch_x = values[0]
#             else:
#                 batch_x = values
#
#         batch_x = [x.to(device) for x in batch_x]
#         batch_probs, gating_probs = model(batch_x)
#
#         batch_probs = batch_probs.detach().cpu().numpy()
#         gating_probs = gating_probs.detach().cpu().numpy()
#         batch_y = batch_y.detach().cpu().numpy()
#
#         valid_mask = batch_y != -1
#
#         if np.sum(valid_mask) == 0:
#             continue  # skip batch
#
#         batch_probs = batch_probs[valid_mask]
#         gating_probs = gating_probs[valid_mask]
#         batch_y = batch_y[valid_mask]
#         batch_identifiers = [utt_id[i] for i in range(len(utt_id)) if valid_mask[i]]
#
#         if total_gating_weights is None:
#             total_gating_weights = np.sum(gating_probs, axis=0)
#         else:
#             total_gating_weights += np.sum(gating_probs, axis=0)
#
#         batch_count += len(batch_y)
#
#         averaged_gating_weights = total_gating_weights / batch_count  # shape: (num_experts)
#         np.save(npy_path, averaged_gating_weights)
#
#         scores.extend(batch_probs)
#         labels.extend(batch_y)
#         identifiers.extend(batch_identifiers)
#
#     with open(save_path, 'w') as fh:
#         for f, pred, lab in zip(identifiers, scores, labels):
#             fh.write('{} {} {}\n'.format(f, pred, lab))
#
#     print('Scores saved to {}\n'.format(save_path))
#     return scores, labels

#-------------------------------- BASELINE ---------------------------------#

def train_epoch_moe(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = [x.to(device) for x in batch_x]
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

        batch_x = [x.to(device) for x in batch_x]
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
