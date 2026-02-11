import os
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import torch
from utils.make_artifacts import *
from scipy.ndimage import gaussian_filter


MANIPULATIONS_REGISTRY = {
    "LSTM": "f0_shift",
    "BILSTM": "f0_shift",
    "GRU": "f0_shift",
}

#--------------------------- COMMON FUNCTIONS ------------------------#

def prepare_audio_window(x, max_len, is_eval=False):
    x_len = x.shape[0]

    while x_len < max_len:
        num_samples = min(x_len, max_len - x_len)
        # x = np.concatenate([x, x[:num_samples]], axis=0)
        x = torch.cat([x, x[:num_samples]], dim=0)
        x_len = x.shape[0]

    if not is_eval:
        last_valid_start_sample = x_len - max_len
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_pad = x[start_sample: start_sample + max_len]

    else:
        # TEST ON THE MIDDLE WINDOW
        num_windows = round(x_len / max_len)
        if num_windows % 2 == 1:  # odd no. of windows
            middle = num_windows//2
        else:  # even no. of windows
            middle = (num_windows//2) - 1

        start_sample = middle * max_len
        x_pad = x[start_sample: start_sample + max_len]

    return x_pad

def load_audio_track(path, trim=False, normalize=False, sr=16000, top_db=40):
    waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    waveform = torch.Tensor(waveform).unsqueeze(0)  # gotta fix torchaudio.load

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)

    # Normalize waveform
    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    # Trim leading and trailing silence
    if trim:
        amplitude_threshold = 10 ** (-top_db / 20)
        non_silent = (waveform.abs() > amplitude_threshold).nonzero(as_tuple=True)[1]
        if non_silent.numel() > 0:
            start = non_silent[0].item()
            end = non_silent[-1].item() + 1
            waveform = waveform[:, start:end]
        else:
            waveform = waveform[:, :0]

    return waveform.squeeze()

class STFTWindowParams:
    def __init__(self, sample_rate=16000, win_sec=0.032, hop_sec=0.016):
        self.sample_rate = sample_rate
        self.win_sec = win_sec
        self.hop_sec = hop_sec

        self.update()

    def update(self):
        self.win_size = int(self.win_sec * self.sample_rate)
        self.hop_size = int(self.hop_sec * self.sample_rate)
        self.fft_size = self.next_pow_2(self.win_size)

    @staticmethod
    def next_pow_2(x):
        return 2 ** (x - 1).bit_length()

def extract_feature(x, mod, stft_params):

    if mod in ['f0_shift', 'f0_shuffle']:
        feat = extract_f0(
            x,
            fs=stft_params.sample_rate,
            win_size=stft_params.win_size,
            hop_size=stft_params.hop_size
        )
        feat = np.nan_to_num(feat, nan=0.0)

    else:
        feat = None
        print(f"Mod not recognized.")

    return feat

def apply_manipulation(feat, mod, stft_params):

    if mod in ['f0_shift', 'f0_shuffle']:
        feat_mod = apply_f0_mod(feat, mod=mod, hop_s=stft_params.hop_sec)
    else:
        feat_mod = None
        print("Not recognized: ", mod)

    return feat_mod

#----------------------------- DATALOADERS  --------------------------#

class LoadTrainData(Dataset):
    def __init__(self, list_IDs, config):
        self.list_IDs = list_IDs

        self.mod = config["mod"]

        self.fs = config['fs']
        self.win_len = config['win_len']
        self.stft_params = STFTWindowParams()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        path = self.list_IDs[idx]

        filename = os.path.splitext(os.path.basename(path))[0]
        file = f'/home/smg/v-viola/moe_antideepfake/temp_feat/f0_asv19/{filename}.npy'

        if not os.path.exists(file):
            x = load_audio_track(path, trim=True, normalize=True)
            x = torch.Tensor(x)
            x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=False)

            feat = extract_feature(x, self.mod, self.stft_params)
            np.save(file, feat)
        else:
            feat = np.load(file)

        label = 0

        if np.all(feat == 0.0):
            label = 1 if np.random.rand() < 0.5 else 0
            feat = torch.tensor(feat, dtype=torch.float32)
            return feat, label

        if idx % 2 == 0:

            feat_mod = apply_manipulation(feat, mod=self.mod, stft_params=self.stft_params)
            feat = feat_mod
            label = 1

        feat = torch.tensor(feat, dtype=torch.float32)
        return feat, label

class LoadTrainData_v2(Dataset):
    def __init__(self, list_IDs, models_list, config):
        self.list_IDs = list_IDs

        self.models_list = models_list

        self.fs = config['fs']
        self.win_len = config['win_len']
        self.stft_params = STFTWindowParams()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        path = self.list_IDs[idx]
        x = load_audio_track(path, trim=True, normalize=True)
        x = torch.Tensor(x)
        x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=False)

        label = 0
        feat_list = []

        for model_type in self.models_list:
            manip_type = MANIPULATIONS_REGISTRY[model_type]

            feat = extract_feature(x, manip_type, self.stft_params)

            # if the feature is corrupted somehow -> shuffle the label
            # TODO: now ok for f0, to be made more general
            if np.all(feat == 0.0):
                label = 1 if np.random.rand() < 0.5 else 0

            elif idx % 2 == 0:
                feat_mod = apply_manipulation(feat, mod=manip_type, stft_params=self.stft_params)
                feat = feat_mod
                label = 1

            feat = torch.tensor(feat, dtype=torch.float32)
            feat_list.append(feat)

        return feat_list, label


#------------------------------ PoC STUFF ----------------------------#

class Load_POC_Data(Dataset):
    def __init__(self, list_IDs, config, multi_band=False, is_eval=False):
        self.list_IDs = list_IDs
        self.win_len = config['win_len']
        self.fs = config['fs']
        self.win_len_samples = int(self.win_len * self.fs)
        self.multi_band = multi_band
        self.is_eval = is_eval

        self.bands = config.get("bands", [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000)])
        self.f_low, self.f_high = config.get("band", (0, 8000))

        self.p_perturb = config.get("p_perturb", 0.5)
        self.snr_db = config.get("snr_db", 30)
        self.n_fft = config.get("n_fft", 512)
        self.hop = config.get("hop", 256)

    def __len__(self):
        return len(self.list_IDs)

    def add_band_limited_noise(self, mag, band):
        f_low, f_high = band
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        freq_idx = torch.where((freqs >= f_low) & (freqs <= f_high))[0]

        noise = torch.randn_like(mag)
        # signal_power = torch.mean(mag[freq_idx, :] ** 2) + 1e-8
        noise_power = torch.mean(noise[freq_idx, :] ** 2) + 1e-8
        alpha = torch.sqrt(1.0 / (noise_power * 10 ** (self.snr_db / 10)))

        band_mask = torch.zeros_like(mag)
        band_mask[freq_idx, :] = 1.0

        mag_noisy = mag + alpha * noise * band_mask
        return mag_noisy

# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(5, 3))
# plt.imshow(librosa.amplitude_to_db(mag_noisy.numpy(), ref=np.max), origin='lower', aspect='auto')
# plt.show()

    def __getitem__(self, idx):
        path = self.list_IDs[idx]
        x, _ = librosa.load(path, sr=self.fs)
        x = torch.Tensor(x)
        x = prepare_audio_window(x, self.win_len_samples, is_eval=False)

        window = torch.hann_window(self.n_fft, device=x.device)
        X = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=window,
            center=False, return_complex=True
        )
        mag = torch.abs(X)

        key = os.path.splitext(os.path.basename(path))[0]
        label = 0
        band_idx = 0

        if not self.is_eval:
            if random.random() < self.p_perturb:
                if self.multi_band:
                    band_idx = random.randrange(len(self.bands))
                    band = self.bands[band_idx]
                    mag = self.add_band_limited_noise(mag, band)
                else:
                    mag = self.add_band_limited_noise(mag, (self.f_low, self.f_high))

                label = 1
            mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
            # return mag, label
            return mag, label, band_idx

        else:
            mag = self.add_band_limited_noise(mag, (self.f_low, self.f_high))
            mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
            return mag, label, key

        # Shortcut 4 real data, comment if-else above
        mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
        return mag, label, key

class LoadTrainData_Pitch(Dataset):
    def __init__(self, list_IDs, config):
        self.list_IDs = list_IDs

        # --- base parameters (Hz, seconds) ---
        self.fs = config['fs']
        self.win_len = config['win_len']

        # --- STFT window params ---
        self.stft_params = STFTWindowParams()

        self.window = torch.hann_window(self.stft_params.fft_size, device='cpu')
        self.p_perturb = config.get("p_perturb", 0.6)

    def __len__(self):
        return len(self.list_IDs)

    def extract_f0(self, waveform,
                      fs, win_size, hop_size,
                      fmin: int = 70, fmax: int = 400) -> ndarray:

        x = np.copy(waveform)
        x = librosa.effects.preemphasis(x)
        autocorr_win_len = win_size // 2

        f0_freqs, _, _ = librosa.pyin(
            x, sr=fs,
            frame_length=win_size, hop_length=hop_size, center=True,
            win_length=autocorr_win_len, fmin=fmin, fmax=fmax, n_thresholds=7
        )

        return f0_freqs

    def flatten_f0(self, f0_traj, strength=0.9, p=0.8):

        f0_mod = f0_traj.copy()
        voiced = ~np.isnan(f0_mod)

        f0_ref = np.nanmedian(f0_mod)
        mask = (np.random.rand(len(f0_mod)) < p) & voiced

        f0_mod[mask] = (
                (1.0 - strength) * f0_mod[mask]
                + strength * f0_ref
        )

        return f0_mod

    def pitch_shift_f0(self, f0_traj, shift_factor):
        f0_mod = f0_traj.copy()
        voiced = ~np.isnan(f0_mod)
        f0_mod[voiced] *= shift_factor
        return f0_mod

    def partial_pitch_shift_f0(self, f0_traj, shift_factor, p=0.5):
        f0_mod = f0_traj.copy()
        voiced = ~np.isnan(f0_mod)
        mask = (np.random.rand(len(f0_mod)) < p) & voiced
        f0_mod[mask] *= shift_factor
        return f0_mod

    def to_TF_plane(self, traj, fs, n_fft):
        frequency_resolution = fs / n_fft

        time_idxs = np.argwhere(np.isnan(traj.T) == False)
        freq_idxs = (traj[time_idxs] / frequency_resolution).astype(int)

        out_contour_matrix = np.zeros((n_fft // 2 + 1, traj.shape[0]))
        out_contour_matrix[freq_idxs, time_idxs] = np.ones(time_idxs.shape)

        return out_contour_matrix

    def __getitem__(self, idx):
        path = self.list_IDs[idx]
        x, _ = librosa.load(path, sr=self.fs)
        x = torch.Tensor(x)
        x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=False)

        X = torch.stft(
            x,
            n_fft=self.stft_params.fft_size,
            hop_length=self.stft_params.hop_size,
            win_length=self.stft_params.win_size,
            window=self.window,
            center=True,
            return_complex=True
        )

        mag = torch.abs(X)

        label = 0

        if random.random() < self.p_perturb:
            f0_traj = self.extract_f0(
                x.numpy(),
                fs=self.stft_params.sample_rate,
                win_size=self.stft_params.win_size,
                hop_size=self.stft_params.hop_size
            )

            if np.all(np.isnan(f0_traj)):
                label = 1 if np.random.rand() < 0.5 else 0
                return torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0), label

            r = random.random()

            partial_prob = 0.5
            flatten_prob = 0.4

            if r < partial_prob:
                # --- partial pitch shift ---
                shift = random.choice([
                    random.uniform(0.3, 0.8),
                    random.uniform(1.5, 2)
                ])
                f0_traj_mod = self.partial_pitch_shift_f0(
                    f0_traj,
                    shift_factor=shift,
                    p=random.uniform(0.3, 0.8)
                )

            elif r < partial_prob + flatten_prob:
                # --- f0 flattening ---
                f0_traj_mod = self.flatten_f0(
                    f0_traj,
                    # strength=random.uniform(0.6, 0.9),
                    p=random.uniform(0.5, 0.9)
                )

            else:
                # --- full pitch shift ---
                shift = random.choice([
                    random.uniform(0.3, 0.8),
                    random.uniform(1.5, 2)
                ])
                f0_traj_mod = self.pitch_shift_f0(
                    f0_traj,
                    shift_factor=shift
                )

            mask_old = gaussian_filter(self.to_TF_plane(f0_traj, self.fs, self.stft_params.fft_size), sigma=1)[None, :, :]
            mask_new = gaussian_filter(self.to_TF_plane(f0_traj_mod, self.fs, self.stft_params.fft_size), sigma=1)[None, :, :]

            # mask_old = mask_old ** 2
            # mask_new = mask_new ** 2

            att = 1  # attenuation factor, [0, 1]
            emp = 8  # emphasis factor [>= 1]
            mask_old = torch.from_numpy(mask_old).float().to(mag.device)
            mask_new = torch.from_numpy(mask_new).float().to(mag.device)
            mag_new = mag * (1.0 - att * mask_old)
            mag_new = mag_new * (1.0 + emp * mask_new)

            mag_new = torch.clamp(mag_new, min=1e-8)
            mag = mag_new.squeeze()
            label = 1

        mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)

        return mag, label

#---------------------------- BASELINE STUFF -------------------------#

class LoadTrainData_Baseline(Dataset):
    def __init__(self, list_IDs, labels, config):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = config['win_len']
        self.fs = config['fs']
        self.win_len_samples = int(self.win_len * self.fs)

        df = pd.DataFrame(labels.items(), columns=['path', 'label'])
        self.real_list = list(df[df['label'] == 0]['path'])
        self.fake_list = list(df[df['label'] == 1]['path'])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Balance real and fake samples in a batch
        if index % 2 == 0:
            track = random.choice(self.real_list)
        else:
            track = random.choice(self.fake_list)

        x = load_audio_track(track, sr=self.fs)
        x = prepare_audio_window(x, self.win_len_samples, is_eval=False)

        y = self.labels[track]
        return x, y

class LoadEvalData_Baseline(Dataset):
    def __init__(self, list_IDs, labels, config):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = config['win_len']
        self.fs = config['fs']
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]

        x = load_audio_track(track, sr=self.fs)
        x = prepare_audio_window(x, self.win_len_samples, is_eval=True)

        y = self.labels[track]
        return x, y, track
