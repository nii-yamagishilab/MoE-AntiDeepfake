import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, return_emb=False):
        super(LSTM_Model, self).__init__()

        self.return_emb = return_emb

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1) # expected x: (batch, seq_len, 1)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        emb = out[:, -1, :]  # take last timestep
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class BiLSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, return_emb=False):
        super(BiLSTM_Model, self).__init__()

        self.return_emb = return_emb

        self.bilstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2, two directions!

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.bilstm(x)  # (batch, seq_len, hidden_size*2)
        emb = out[:, -1, :]  # take last timestep
        # emb = torch.mean(out, dim=1)
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class GRU_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, return_emb=False):
        super(GRU_Model, self).__init__()

        self.return_emb = return_emb

        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        emb = out[:, -1, :]  # take last timestep
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb