import torch
from tqdm import tqdm
import numpy as np
import os


def smooth_labels(y, eps=0.1):
    return y * (1 - eps) + 0.5 * eps

def train_epoch_expert(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_total = 0.0
    model.train()

    for batch_x, batch_y, _ in tqdm(train_loader, total=len(train_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)

        batch_y = smooth_labels(batch_y, eps=0.1)

        batch_out, batch_emb = model(batch_x)

        loss_bce = criterion(batch_out, batch_y)
        # loss_supcon = supconloss(batch_emb, batch_y.long())

        # batch_loss = loss_bce + (0.5 * loss_supcon)
        batch_loss = loss_bce

        batch_size = batch_x.size(0)
        running_loss += (batch_loss.item() * batch_size)
        num_total += batch_size

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss

def valid_epoch_expert(data_loader, model, criterion, device):
    running_loss = 0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y, _ in tqdm(data_loader, total=len(data_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)
        batch_out, batch_emb = model(batch_x)

        loss_bce = criterion(batch_out, batch_y)
        # loss_supcon = supconloss(batch_emb, batch_y.long())

        # batch_loss = loss_bce + (0.5 * loss_supcon)
        batch_loss = loss_bce

        batch_size = batch_x.size(0)
        running_loss += (batch_loss.item() * batch_size)
        num_total += batch_size

    running_loss /= num_total
    return running_loss

def eval_expert(model, data_loader, device):
    model.eval()
    fname_list = []
    score_list = []
    label_list = []

    total_removed = 0
    total_seen = 0

    for batch_x, batch_y, utt_id, _ in tqdm(data_loader, total=len(data_loader)):
        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            if len(values) == 1:
                batch_x = values[0]
            else:
                batch_x = values

        batch_x = torch.stack([x.to(device) for x in batch_x])

        batch_out = model(batch_x)

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

    return fname_list, score_list, label_list

def expert_loop_calibration(data_loader, expert, device, is_test=False):
    expert.eval()

    logits_list = []
    label_list = []
    batch_k_list = []
    batch_a_list = []
    total_removed = 0
    total_seen = 0

    all_emb_r = []
    all_emb_f = []
    all_emb = []

    with torch.no_grad():
        for batch_x, batch_y, batch_k, batch_a in tqdm(data_loader, total=len(data_loader)):
            if isinstance(batch_x, dict):
                values = list(batch_x.values())
                if len(values) == 1:
                    batch_x = values[0]
                else:
                    batch_x = values

            batch_x = torch.stack([x.to(device) for x in batch_x])
            batch_logits, embeddings = expert(batch_x)

            batch_logits = batch_logits.cpu().numpy().ravel()
            embeddings = embeddings.cpu().numpy()
            batch_y = np.array(batch_y)

            valid_mask = batch_y != -1
            total_removed += np.sum(~valid_mask)
            total_seen += len(batch_y)
            if not np.any(valid_mask):
                continue

            batch_logits = batch_logits[valid_mask]
            batch_y = batch_y[valid_mask]

            logits_list.extend(batch_logits.tolist())
            label_list.extend(batch_y.tolist())
            batch_k_list.extend(np.array(batch_k)[valid_mask].tolist())
            batch_a_list.extend(np.array(batch_a)[valid_mask].tolist())

            embeddings = embeddings[valid_mask]
            emb_real = embeddings[batch_y == 0]
            emb_fake = embeddings[batch_y == 1]

            all_emb.append(embeddings)
            all_emb_r.append(emb_real)
            all_emb_f.append(emb_fake)

    all_emb_r = np.vstack(all_emb_r)
    all_emb_f = np.vstack(all_emb_f)
    all_emb = np.vstack(all_emb)

    print(f"Removed {total_removed} corrupted samples out of {total_seen} total "
          f"({100 * total_removed / total_seen:.2f}%).")

    if is_test:
        return logits_list, label_list, all_emb, batch_k_list, batch_a_list
    else:
        return logits_list, label_list, all_emb_r, all_emb_f

#------------------------------------ NOT IN USE - MOE ----------------------------------#

def train_epoch_compound(train_loader, model, optim, losses, config, device):
    num_tot = 0.0
    num_tot_fakes = 0.0
    num_tot_real = 0.0

    running_bce_loss = 0.0
    running_attr_loss = 0.0
    running_tot_loss = 0.0

    model.train()
    if config['freeze_experts']:
        for expert in model.experts:
            expert.eval()

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
        batch_probs = batch_probs.clamp(1e-8, 1 - 1e-8)
        gate_probs = gate_probs.clamp(1e-8, 1 - 1e-8)

        # BCELoss (Real vs Fake)
        batch_bce_loss = losses['bce'](batch_probs, batch_y.unsqueeze(-1))

        # NLLLoss (Attribution Loss, fakes-only)
        log_gate_probs = torch.log(gate_probs + 1e-8)
        if fake_mask.any():
            batch_attr_loss = losses['nll'](log_gate_probs[fake_mask],
                                            batch_k[fake_mask])
        else:
            batch_attr_loss = torch.tensor(0.0, device=batch_x.device)

        running_bce_loss += batch_bce_loss.item() * batch_size
        running_attr_loss += batch_attr_loss.item() * fake_mask.sum().item()

        batch_total_loss = (config['wb']*batch_bce_loss +
                            config['wa']*batch_attr_loss)

        running_tot_loss += batch_total_loss.item() * batch_size

        optim.zero_grad()
        batch_total_loss.backward()
        optim.step()

    running_bce_loss /= num_tot
    running_attr_loss /= max(num_tot_fakes, 1)
    running_tot_loss /= num_tot

    epoch_losses = {
        "total": running_tot_loss,
        "bce": running_bce_loss,
        "attr": running_attr_loss,
    }
    return epoch_losses

def valid_epoch_compound(data_loader, model, losses, config, device):
    num_tot = 0.0
    num_tot_fakes = 0.0
    num_tot_real = 0.0

    running_bce_loss = 0.0
    running_attr_loss = 0.0
    running_tot_loss = 0.0

    model.eval()

    scores = []
    labels = []

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
                                            batch_k[fake_mask])
        else:
            batch_attr_loss = torch.tensor(0.0, device=batch_x.device)

        running_bce_loss += batch_bce_loss.item() * batch_size
        running_attr_loss += batch_attr_loss.item() * fake_mask.sum().item()

        batch_total_loss = (batch_bce_loss +
                            config['wa'] * batch_attr_loss)

        running_tot_loss += batch_total_loss.item() * batch_size

        batch_probs = batch_probs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        valid_mask = batch_y != -1
        if np.sum(valid_mask) == 0:
            continue
        batch_probs = batch_probs[valid_mask]
        batch_y = batch_y[valid_mask]
        scores.extend(batch_probs)
        labels.extend(batch_y)

    running_bce_loss /= num_tot
    running_attr_loss /= max(num_tot_fakes, 1)
    running_tot_loss /= num_tot

    epoch_losses = {
        "total": running_tot_loss,
        "bce": running_bce_loss,
        "attr": running_attr_loss,
    }

    scores = np.array(scores)
    labels = np.array(labels)

    return epoch_losses, scores, labels

def eval_moe_compound(model, data_loader, save_dir, device):
    model.eval()

    # ---- CREATE DIRECTORY IF NEEDED ----
    os.makedirs(save_dir, exist_ok=True)

    # ---- SCORE FILE ----
    score_txt_path = os.path.join(save_dir, "scores.txt")

    # ---- GATING PATHS ----
    npy_gating_global = os.path.join(save_dir, "gating_mean_global.npy")
    npy_gating_fake = os.path.join(save_dir, "gating_mean_fake.npy")
    npy_gating_real = os.path.join(save_dir, "gating_mean_real.npy")
    npy_gating_all = os.path.join(save_dir, "gating_all.npy")

    # ---- WEIGHTED EXPERT PATHS ----
    npy_weighted_global = os.path.join(save_dir, "weighted_mean_global.npy")
    npy_weighted_fake = os.path.join(save_dir, "weighted_mean_fake.npy")
    npy_weighted_real = os.path.join(save_dir, "weighted_mean_real.npy")
    npy_weighted_all = os.path.join(save_dir, "weighted_all.npy")

    # ---- EXPERT PROBS PATHS ----
    npy_expert_global = os.path.join(save_dir, "expert_mean_global.npy")
    npy_expert_fake = os.path.join(save_dir, "expert_mean_fake.npy")
    npy_expert_real = os.path.join(save_dir, "expert_mean_real.npy")
    npy_expert_all = os.path.join(save_dir, "expert_all.npy")

    all_paths = [
        npy_gating_global, npy_gating_fake, npy_gating_real, npy_gating_all,
        npy_weighted_global, npy_weighted_fake, npy_weighted_real, npy_weighted_all,
        npy_expert_global, npy_expert_fake, npy_expert_real, npy_expert_all
    ]

    # ---- REMOVE OLD FILES ----
    for p in all_paths + [score_txt_path]:
        if os.path.exists(p):
            os.remove(p)
            print(f'[INFO] Removed existing file: {p}')
    print()

    scores = []
    labels = []
    identifiers = []

    all_weights = []
    all_weighted = []
    all_experts = []

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):

        if isinstance(batch_x, dict):
            values = list(batch_x.values())
            batch_x = values[0] if len(values) == 1 else values

        batch_x = [x.to(device) for x in batch_x]

        # ---- MODEL OUTPUTS ----
        batch_probs, gating_probs, weighted_expert_probs, expert_probs = model(batch_x)
        # print(expert_probs.shape)
        # print(expert_probs[:, 1])
        # # print()
        # # print(calib_probs.shape)
        # # print(calib_probs[:, 0])
        # # print()
        # print(utt_id)
        # print()
        # import sys
        # sys.exit()

        batch_probs = batch_probs.detach().cpu().numpy()
        gating_probs = gating_probs.detach().cpu().numpy()
        weighted_expert_probs = weighted_expert_probs.detach().cpu().numpy()
        expert_probs = expert_probs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()

        # ---- FILTER INVALID ----
        valid_mask = batch_y != -1
        if np.sum(valid_mask) == 0:
            continue

        batch_probs = batch_probs[valid_mask]
        gating_probs = gating_probs[valid_mask]
        weighted_expert_probs = weighted_expert_probs[valid_mask]
        expert_probs = expert_probs[valid_mask]
        batch_y = batch_y[valid_mask]

        batch_identifiers = [utt_id[i] for i in range(len(utt_id)) if valid_mask[i]]

        scores.extend(batch_probs)
        labels.extend(batch_y)
        identifiers.extend(batch_identifiers)

        all_weights.extend(gating_probs)
        all_weighted.extend(weighted_expert_probs)
        all_experts.extend(expert_probs)

    scores = np.array(scores)
    labels = np.array(labels)

    all_weights = np.array(all_weights)
    all_weighted = np.array(all_weighted)
    all_experts = np.array(all_experts)

    # =========================================================
    # ---------------- GATING STATS ---------------------------
    # =========================================================
    np.save(npy_gating_global, all_weights.mean(axis=0))
    np.save(npy_gating_all, all_weights)

    if (labels == 1).sum() > 0:
        np.save(npy_gating_fake, all_weights[labels == 1].mean(axis=0))
    else:
        print("No fake samples found for gating.")

    if (labels == 0).sum() > 0:
        np.save(npy_gating_real, all_weights[labels == 0].mean(axis=0))
    else:
        print("No real samples found for gating.")

    # =========================================================
    # ------------ WEIGHTED EXPERT STATS ----------------------
    # =========================================================
    np.save(npy_weighted_global, all_weighted.mean(axis=0))
    np.save(npy_weighted_all, all_weighted)

    if (labels == 1).sum() > 0:
        np.save(npy_weighted_fake, all_weighted[labels == 1].mean(axis=0))
    else:
        print("No fake samples found for weighted experts.")

    if (labels == 0).sum() > 0:
        np.save(npy_weighted_real, all_weighted[labels == 0].mean(axis=0))
    else:
        print("No real samples found for weighted experts.")

    # =========================================================
    # ---------------- EXPERT STATS ---------------------------
    # =========================================================
    np.save(npy_expert_global, all_experts.mean(axis=0))
    np.save(npy_expert_all, all_experts)

    if (labels == 1).sum() > 0:
        np.save(npy_expert_fake, all_experts[labels == 1].mean(axis=0))
    else:
        print("No fake samples found for experts.")

    if (labels == 0).sum() > 0:
        np.save(npy_expert_real, all_experts[labels == 0].mean(axis=0))
    else:
        print("No real samples found for experts.")

    # ---- SAVE SCORES FILE ----
    with open(score_txt_path, 'w') as fh:
        for f, pred, lab in zip(identifiers, scores, labels):
            fh.write(f'{f} {pred} {lab}\n')

    print(f'\nScores saved to {score_txt_path}')
    print("Saved gating, weighted experts, and expert statistics.\n")

    return scores, labels


#-------------------------------- NOT IN USE - BASELINE ---------------------------------#

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
