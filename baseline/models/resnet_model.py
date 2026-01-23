import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision
from torchvision.models import resnet18


def modify_for_grayscale(model):
    first_conv_layer = model.conv1

    new_first_conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )
    with torch.no_grad():
        new_first_conv_layer.weight[:, 0] = first_conv_layer.weight.mean(dim=1)
        if first_conv_layer.bias is not None:
            new_first_conv_layer.bias = first_conv_layer.bias

    model.conv1 = new_first_conv_layer # resnet18
    return model


class ResNet_LogSpec(nn.Module):
    def __init__(self, sample_rate=16000, return_emb=False, num_class=2):
        super(ResNet_LogSpec, self).__init__()

        self.return_emb = return_emb

        if sample_rate == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
        elif sample_rate == 22050:
            n_fft = 704
            win_length = 552
            hop_length = 220
        elif sample_rate == 24000:
            n_fft = 768
            win_length = 600
            hop_length = 240
        else:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=self.n_fft,
                                  win_length=self.win_length,
                                  hop_length=self.hop_length,
                                  power=2, window_fn=torch.hamming_window)

        self.model = resnet18(pretrained=False)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.embedding_layer = nn.Linear(num_ftrs, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_class)
        self.softmax = nn.Softmax(dim=1)

        self.to_db = T.AmplitudeToDB()
        # self.normalize = torchvision.transforms.Normalize(mean=0.449, std=0.226)

        self.latent_dim = 256

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stft(x)
        x = self.to_db(x)
        # x = self.normalize(x)

        x = self.model(x)
        x = self.dropout(x)
        emb = self.embedding_layer(x)
        x = self.relu(emb)
        logits = self.classifier(x)
        out = self.softmax(logits)

        if self.return_emb:
            return out, emb
        else:
            return out

class ResNet_MelSpec(nn.Module):
    def __init__(self, sample_rate=16000, return_emb=False, num_class=2):
        super(ResNet_MelSpec, self).__init__()

        self.return_emb = return_emb

        if sample_rate == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            f_max = 7600
        elif sample_rate == 22050:
            n_fft = 704
            win_length = 552
            hop_length = 220
            f_max = 10500
        elif sample_rate == 24000:
            n_fft = 768
            win_length = 600
            hop_length = 240
            f_max = 11000
        else:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=20, f_max=f_max,
            n_mels=80, window_fn=torch.hamming_window
        )

        self.model = resnet18(pretrained=False)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.embedding_layer = nn.Linear(num_ftrs, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, num_class)
        self.softmax = nn.Softmax(dim=1)

        self.to_db = torchaudio.transforms.AmplitudeToDB()
        # self.normalize = torchvision.transforms.Normalize(mean=0.449,std=0.226)

        self.latent_dim = 256

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.melspectrogram(x)
        x = self.to_db(x)
        # x = self.normalize(x)

        x = self.model(x)
        x = self.dropout(x)
        emb = self.embedding_layer(x)
        x = self.relu(emb)
        logits = self.classifier(x)
        out = self.softmax(logits)

        if self.return_emb:
            return out, emb
        else:
            return out