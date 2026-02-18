import torch.nn as nn


INPUT_SIZE = 257


class LSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=INPUT_SIZE, return_emb=False):
        super(LSTM_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)  # expected x: (batch, seq_len, n_feat)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        emb = out[:, -1, :]  # take last timestep
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class BiLSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=INPUT_SIZE, return_emb=False):
        super(BiLSTM_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size

        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2, two directions!

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)  # expected x: (batch, seq_len, n_feat)
        out, _ = self.bilstm(x)  # (batch, seq_len, hidden_size*2)
        emb = out[:, -1, :]  # take last timestep
        # emb = torch.mean(out, dim=1)
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class GRU_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=INPUT_SIZE, return_emb=False):
        super(GRU_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)  # expected x: (batch, seq_len, n_feat)
        out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        emb = out[:, -1, :]  # take last timestep
        out = self.fc(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class MLP_Model(nn.Module):
    def __init__(self, hidden_size=16, input_size=10, return_emb=False):
        super(MLP_Model, self).__init__()
        self.return_emb = return_emb

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)
        emb = self.relu(self.fc1(x))
        out = self.fc2(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class MLP_Model_v2(nn.Module):
    def __init__(self, input_size=21, return_emb=False):
        super(MLP_Model_v2, self).__init__()
        self.return_emb = return_emb

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)
        x = self.relu(self.fc1(x))
        emb = self.relu(self.fc2(x))
        out = self.fc3(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb