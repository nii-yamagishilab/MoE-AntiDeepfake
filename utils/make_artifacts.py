import numpy as np
import librosa
from torch.utils.data import Dataset
from numpy import ndarray
import torch
import torchaudio
from utils.data_utils import prepare_audio_window
import random
from scipy.ndimage import gaussian_filter
import os


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

# # USAGE:
# stft_params = STFTWindowParams()
# print(stft_params.win_size, stft_params.hop_size, stft_params.fft_size)

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
            emp = 2  # emphasis factor [>= 1]
            mask_old = torch.from_numpy(mask_old).float().to(mag.device)
            mask_new = torch.from_numpy(mask_new).float().to(mag.device)
            mag_new = mag * (1.0 - att * mask_old)
            mag_new = mag_new * (1.0 + emp * mask_new)

            mag_new = torch.clamp(mag_new, min=1e-8)
            mag = mag_new.squeeze()
            label = 1

        mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)

        return mag, label



# import matplotlib.pyplot as plt
#
# plt.imshow(mask_old.squeeze(), origin='lower', aspect='auto')
# plt.show()
# plt.imshow(mask_new.squeeze(), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow(mag, origin='lower', aspect='auto')
# plt.show()
# plt.imshow(mag_new.squeeze(), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow((mag-mag_new.squeeze()), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow(torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0), origin='lower', aspect='auto')
# plt.show()
# plt.imshow(torchaudio.functional.amplitude_to_DB(mag_new.squeeze(), multiplier=20.0, amin=1e-10, db_multiplier=1.0), origin='lower', aspect='auto')
# plt.show()
#
# ciao = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0) - torchaudio.functional.amplitude_to_DB(mag_new.squeeze(), multiplier=20.0, amin=1e-10, db_multiplier=1.0)
# plt.imshow(ciao, origin='lower', aspect='auto')
# plt.show()


class LoadTrainData_Pitch_DEBUG(Dataset):
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
        x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=True)

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

        f0_traj = self.extract_f0(
            x.numpy(),
            fs=self.stft_params.sample_rate,
            win_size=self.stft_params.win_size,
            hop_size=self.stft_params.hop_size
        )

        if np.all(np.isnan(f0_traj)):
            return torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0), label

        f0_traj_mod = self.flatten_f0(
            f0_traj,
            # strength=random.uniform(0.6, 0.9),
            p=random.uniform(0.5, 0.9)
        )

        mask_old = gaussian_filter(self.to_TF_plane(f0_traj, self.fs, self.stft_params.fft_size), sigma=1)[None, :, :]
        mask_new = gaussian_filter(self.to_TF_plane(f0_traj_mod, self.fs, self.stft_params.fft_size), sigma=1)[None, :, :]

        att = 1  # attenuation factor, [0, 1]
        emp = 2  # emphasis factor [>= 1]
        mask_old = torch.from_numpy(mask_old).float().to(mag.device)
        mask_new = torch.from_numpy(mask_new).float().to(mag.device)
        mag_new = mag * (1.0 - att * mask_old)
        mag_new = mag_new * (1.0 + emp * mask_new)

        mag_new = torch.clamp(mag_new, min=1e-8)
        mag = mag_new.squeeze()

        track_name = os.path.splitext(os.path.basename(path))[0]

        rootpath = '/home/smg/v-viola/moe_antideepfake/examples/pitch'
        mod_dir = 'f0_flat'
        np.save(os.path.join(rootpath, mod_dir, track_name + '.npy'), mag)

        # mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)

        return mag, label