import torch.nn as nn


class GRU_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=21, return_emb=False):
        super(GRU_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size
        self.latent_dim = hidden_size

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.latent_dim,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.latent_dim, 1)

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


class MLP_Smooth(nn.Module):
    def __init__(self, input_size=13, return_emb=False):
        super(MLP_Smooth, self).__init__()
        self.return_emb = return_emb

        self.input_size = input_size
        self.latent_dim = 16

        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 1)
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


class MLP_Flat_Pitch(nn.Module):
    def __init__(self, hidden_size=16, input_size=10, return_emb=False):
        super(MLP_Flat_Pitch, self).__init__()
        self.return_emb = return_emb

        self.latent_dim = hidden_size
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, self.latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        if self.input_size == 1:
            x = x.unsqueeze(-1)
        emb = self.relu(self.fc1(x))
        out = self.fc2(emb).squeeze(1)

        if not self.return_emb:
            return out
        else:
            return out, emb


class MLP_F0_Incons(nn.Module):
    def __init__(self, input_size=10, return_emb=False):
        super(MLP_F0_Incons, self).__init__()
        self.return_emb = return_emb

        self.input_size = input_size
        self.latent_dim = 16

        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 1)
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


class MLP_Energy(nn.Module):
    def __init__(self, input_size=24, return_emb=False):
        super(MLP_Energy, self).__init__()
        self.return_emb = return_emb

        self.input_size = input_size
        self.latent_dim = 16

        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 1)
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

class MLP_Checkerboard(nn.Module):
    def __init__(self, input_size=225, return_emb=False):
        super(MLP_Checkerboard, self).__init__()
        self.return_emb = return_emb

        self.input_size = input_size
        self.latent_dim = 32

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 1)
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

# ---------------------------------------- Unused ---------------------------------------- #

class CNN_1D_Model(nn.Module):
    def __init__(self, return_emb=False):
        super(CNN_1D_Model, self).__init__()

        self.return_emb = return_emb

        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.adaptive_avg_pool(x)
        emb = self.flatten(x)
        x = self.linear(emb).squeeze(1)

        if self.return_emb:
            return x, emb
        else:
            return x

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=257, return_emb=False):
        super(LSTM_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size
        self.latent_dim = hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.latent_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.latent_dim, 1)

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
    def __init__(self, hidden_size=64, num_layers=1, input_size=257, return_emb=False):
        super(BiLSTM_Model, self).__init__()

        self.return_emb = return_emb
        self.input_size = input_size
        self.latent_dim = hidden_size

        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.latent_dim,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(self.latent_dim * 2, 1)  # *2, two directions!

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