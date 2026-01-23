import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(train_loader, model, optim, criterion, device):
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


def valid_epoch(data_loader, model, criterion, device):
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
        batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel()

        fname_list.extend(utt_id)
        pred_list.extend(batch_score.tolist())
        label_list.extend(batch_y.tolist())

        with open(save_path, 'w') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{} {} {}\n'.format(f, pred, lab))
        fh.close()

    print('Scores saved to {}\n'.format(save_path))


def eval_moe(model, data_loader, save_path, device):
    model.eval()
    total_gating_weights = None
    batch_count = 0

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
        fname_list = []
        pred_list = []
        label_list = []

        batch_x = batch_x.to(device)

        batch_out, gating_weights = model(batch_x)
        batch_out = nn.Softmax(dim=1)(batch_out)
        batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel()

        # if total_gating_weights is None:
        #     total_gating_weights = np.sum(gating_weights.detach().cpu().numpy(), axis=0)
        # else:
        #     total_gating_weights += np.sum(gating_weights.detach().cpu().numpy(), axis=0)
        #     batch_count += gating_weights.shape[0]
        #
        # averaged_gating_weights = total_gating_weights / batch_count  # shape: (num_experts)
        # np.save(save_path.replace('.txt', '.npy'), averaged_gating_weights)

        fname_list.extend(utt_id)
        pred_list.extend(batch_score.tolist())
        label_list.extend(batch_y.tolist())

        with open(save_path, 'w') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{} {} {}\n'.format(f, pred, lab))
        fh.close()

    print('Scores saved to {}'.format(save_path))