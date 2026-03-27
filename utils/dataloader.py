import os
import pandas as pd
from torch.utils.data import Dataset
from utils.make_artifacts import *
from scipy.ndimage import gaussian_filter
from utils.model_utils import MANIPULATIONS_REGISTRY


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

#-------------------------------------- COMMON FUNCTIONS ----------------------------------#

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

def autocorr_decay(x, max_lag=50):
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode='full')
    ac = ac[len(ac)//2:]
    ac /= ac[0] + 1e-8
    return np.mean(ac[1:max_lag])

def extract_proxy_scalars_F0(feat):
    f0v = feat[feat != 0.0]
    if len(f0v) < 3:
        return np.zeros(10, dtype=np.float32)

    df0 = np.diff(f0v)
    ddf0 = np.diff(df0)

    f0_std = np.std(f0v)
    df0_std = np.std(df0)

    f0_ac_decay = autocorr_decay(f0v)
    df0_mean_abs = np.mean(np.abs(df0))
    df0_max_abs = np.max(np.abs(df0))
    df0_p95 = np.percentile(np.abs(df0), 95)
    thr = np.median(np.abs(df0)) * 3
    df0_outlier_ratio = np.mean(np.abs(df0) > thr)
    ddf0_std = np.std(ddf0)
    dyn_ratio = df0_std / (f0_std + 1e-8)
    curv_ratio = ddf0_std / (df0_std + 1e-8)

    return np.array([f0_std, f0_ac_decay, df0_std, df0_mean_abs, df0_max_abs,
                     df0_p95, df0_outlier_ratio, ddf0_std, dyn_ratio, curv_ratio])

def extract_proxy_scalars_energy(spec_db, num_bands=8, drop_threshold_ratio=0.2):
    freq_bins, n_frames = spec_db.shape
    features = []

    E_total = spec_db.sum(axis=0)  # shape: [n_frames]
    features.append(E_total.mean())
    features.append(E_total.std())
    features.append(E_total.max() - E_total.min())

    band_size = freq_bins // num_bands
    for b in range(num_bands):
        start = b * band_size
        end = start + band_size if b < num_bands - 1 else freq_bins
        E_band = spec_db[start:end, :].sum(axis=0)
        features.append(E_band.mean())
        features.append(E_band.std())

    delta_E = np.diff(E_total)
    features.append(delta_E.mean())
    features.append(delta_E.std())
    features.append(delta_E.max() - delta_E.min())

    # Use max-energy as reference to detect relative drops
    threshold = E_total.max() * drop_threshold_ratio
    drop_frames = np.where(E_total < threshold)[0]
    features.append(len(drop_frames))   # how often the signal dips abnormally low

    if len(drop_frames) > 0:
        # Split into contiguous regions to measure duration of each drop
        drops = np.split(drop_frames, np.where(np.diff(drop_frames) != 1)[0]+1)
        avg_drop_len = np.mean([len(d) for d in drops]) # average duration of energy holes
    else:
        avg_drop_len = 0
    features.append(avg_drop_len)

    return np.array(features, dtype=np.float32)

def extract_proxy_scalars_LSF(feat, lsf_order=21):
    feats = []

    spacing = np.diff(feat, axis=1)  # [T x (order-1)]
    feats.append(spacing.mean())
    feats.append(spacing.std())
    feats.append(spacing.min())
    feats.append(spacing.var())

    spacing_dt = np.diff(spacing, axis=0)
    feats.append(np.abs(spacing_dt).mean())
    feats.append(np.abs(spacing_dt).std())

    delta_t = np.diff(feat, axis=0)
    feats.append(np.abs(delta_t).mean())
    feats.append(np.abs(delta_t).std())
    feats.append(np.abs(delta_t).max())

    delta2_t = feat[2:] - 2*feat[1:-1] + feat[:-2]
    feats.append(np.abs(delta2_t).mean())
    feats.append(np.abs(delta2_t).std())

    var_per_coeff = feat.var(axis=0)
    feats.append(var_per_coeff.mean())
    feats.append(var_per_coeff.std())

    return np.array(feats, dtype=np.float32)


#--------------------------------------- WRAPPERS --------------------------------------#

def extract_feature(x, mod, stft_params):

    if mod in ['f0_incons', 'flat_pitch']:
        feat = extract_f0(
            x,
            fs=stft_params.sample_rate,
            win_size=stft_params.win_size,
            hop_size=stft_params.hop_size
        )
        feat = np.nan_to_num(feat, nan=0.0)

    elif mod == 'oversmooth':
        feat = lsf_extraction(x.numpy(), stft_params=stft_params, lsf_order=20)

    elif mod == 'energy':
        feat = spec_extraction(x, stft_params)
        # feat = energy_extraction(x, stft_params)

    elif mod == 'min_phase':
        feat = phase_extraction(x.numpy(), key='unwrapped', stft_params=stft_params)

    else:
        feat = None
        print(f"Mod not recognized.")

    return feat

def apply_manipulation(feat, mod, stft_params):

    if mod in ['f0_incons', 'flat_pitch']:
        feat_mod = apply_f0_mod(feat, mod=mod, hop_s=stft_params.hop_sec)

    elif mod == 'oversmooth':
        feat_mod = lsf_manipulation_smoothing(feat)

    elif mod == 'energy':
        feat_mod = spec_manipulation(feat, stft_params)  # here feat is the raw wav
        # feat_mod = energy_manipulation(feat, stft_params)  # here feat is the raw wav

    elif mod == 'min_phase':
        feat_mod = phase_manipulation(feat.numpy(), key='unwrapped', stft_params=stft_params)  # here feat is the raw wav

    else:
        feat_mod = None
        print("Not recognized: ", mod)

    return feat_mod


#------------------------------------------ DATALOADERS  ------------------------------------------#

class LoadTrainData(Dataset):
    def __init__(self, list_IDs, models_list, config):
        self.list_IDs = list_IDs
        self.fs = config['fs']
        self.win_len = config['win_len']
        self.stft_params = STFTWindowParams()

        self.mod = config['mod']
        self.config = config
        self.models_list = models_list

        self.gt_f0 = os.path.join(config['rootpath'], "temp_feat", f"f0_{config['train_data']}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        path = self.list_IDs[idx]

        if self.config['train_data'] == 'asv19':
            self.train_stats = self.config['train_stats_asv19']
        elif self.config['train_data'] == 'asv5':
            self.train_stats = self.config["train_stats_asv5"]
        else:
            print(f"Not recognized: {self.config['train_data']}. No available normalization data.")

        mod_list = []
        label = 0
        mod_gt = -1

        # find the manipulation corresponding to each expert in the list
        # assume registry is a list of (model, manipulation)
        if self.mod == 'all':
            registry = MANIPULATIONS_REGISTRY.copy()
            for model_type in self.models_list:
                manipulation = None
                for m, mod in registry:
                    if m == model_type:
                        manipulation = mod
                        registry.remove((m, mod))  # remove the used pair to keep mapping one-to-one
                        break

                if manipulation is None:
                    raise ValueError(f"No manipulation found for model {model_type}")

                mod_list.append(manipulation)
        else:
            mod_list.append(self.mod)

        # EXTRACT FEATURES

        x = load_audio_track(path, trim=True, normalize=True)
        x = torch.Tensor(x)
        feat_dict = {}
        invalid_mods = set()

        for mod in mod_list:

            if mod in ['flat_pitch', 'f0_incons']:
                filename = os.path.splitext(os.path.basename(path))[0]
                file = os.path.join(self.gt_f0, f'{filename}.npy')

                if not os.path.exists(file):
                    feat = extract_feature(x, mod, self.stft_params)
                    np.save(file, feat)
                else:
                    feat = np.load(file)
                    feat = np.nan_to_num(feat, nan=0.0)

                if np.all(feat == 0.0):
                    label = 1 if np.random.rand() < 0.5 else 0
                    feat = np.zeros(10)
                    feat = torch.tensor(feat, dtype=torch.float32)
                    invalid_mods.add(mod)

            else:
                x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=False)
                feat = extract_feature(x, mod, self.stft_params)

            feat_dict[mod] = feat

        # APPLY MANIPULATIONS + EXTRACT PROXY FEATURES WHEN NEEDED

        # Case 1: expert training, one feature - one manipulation

        if len(mod_list) == 1:
            feat = feat_dict[mod_list[0]]
            mod = mod_list[0]

            if mod not in invalid_mods:

                if idx % 2 == 0:

                    if mod in ['min_phase', 'energy']:
                        feat_mod = apply_manipulation(x, mod=mod, stft_params=self.stft_params)
                    else:
                        feat_mod = apply_manipulation(feat, mod=mod, stft_params=self.stft_params)

                    feat = torch.tensor(feat_mod, dtype=torch.float32)
                    label = 1

                # Extract proxy features when needed

                if mod in ['f0_incons', 'flat_pitch']:

                    feat = extract_proxy_scalars_F0(feat.numpy() if isinstance(feat, torch.Tensor) else feat)

                    if mod == 'f0_incons':
                        mean = torch.tensor(self.train_stats['mean_incons'])
                        std = torch.tensor(self.train_stats['std_incons'])
                    else:
                        mean = torch.tensor(self.train_stats['mean_flat'])
                        std = torch.tensor(self.train_stats['std_flat'])

                    if np.all(feat == 0.0):
                        label = 1 if np.random.rand() < 0.5 else 0
                        feat = torch.tensor(np.zeros(10), dtype=torch.float32)
                    else:
                        feat = torch.tensor(feat, dtype=torch.float32)
                        feat = (feat - mean) / (std + 1e-8)

                elif mod == 'energy':
                    feat = extract_proxy_scalars_energy(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                    mean = torch.tensor(self.train_stats['mean_energy'])
                    std = torch.tensor(self.train_stats['std_energy'])
                    feat = torch.tensor(feat, dtype=torch.float32)
                    feat = (feat - mean) / (std + 1e-8)

                elif mod == 'oversmooth':
                    feat = extract_proxy_scalars_LSF(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                    feat = torch.tensor(feat, dtype=torch.float32)
                    mean = torch.tensor(self.train_stats['mean_smooth'])
                    std = torch.tensor(self.train_stats['std_smooth'])
                    feat = (feat - mean) / (std + 1e-8)

            else:
                feat = torch.tensor(np.zeros(10), dtype=torch.float32)

            feat_dict[mod] = feat  # overwrite

        # Case 2: gate training, multiple features - multiple manipulation

        else:
            valid_mods = [m for m in mod_list if m not in invalid_mods]
            selected_mods = []

            if len(valid_mods) > 0:
                # n_mods = np.random.randint(1, len(valid_mods) + 1)
                n_mods = 1
                selected_mods = np.random.choice(valid_mods, size=n_mods, replace=False)
                mod_gt = mod_list.index(selected_mods.item())

            if idx % 2 == 0 and len(selected_mods) > 0:

                for mod in selected_mods:
                    feat = feat_dict[mod]

                    if mod in ['min_phase', 'energy']:
                        feat_mod = apply_manipulation(x, mod=mod, stft_params=self.stft_params)
                    else:
                        feat_mod = apply_manipulation(feat, mod=mod, stft_params=self.stft_params)

                    feat_dict[mod] = torch.tensor(feat_mod, dtype=torch.float32)

                label = 1

            # Extract proxy features when needed

            for mod in mod_list:

                if mod in invalid_mods:
                    feat_dict[mod] = torch.zeros(10, dtype=torch.float32)
                    continue

                feat = feat_dict[mod]

                if mod in ['f0_incons', 'flat_pitch']:

                    feat = extract_proxy_scalars_F0(feat.numpy() if isinstance(feat, torch.Tensor) else feat)

                    if mod == 'f0_incons':
                        mean = torch.tensor(self.train_stats['mean_incons'])
                        std = torch.tensor(self.train_stats['std_incons'])
                    else:
                        mean = torch.tensor(self.train_stats['mean_flat'])
                        std = torch.tensor(self.train_stats['std_flat'])

                    if np.all(feat == 0.0):
                        label = 1 if np.random.rand() < 0.5 else 0
                        feat = torch.zeros(10, dtype=torch.float32)
                    else:
                        feat = torch.tensor(feat, dtype=torch.float32)
                        feat = (feat - mean) / (std + 1e-8)


                elif mod == 'energy':
                    feat = extract_proxy_scalars_energy(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                    mean = torch.tensor(self.train_stats['mean_energy'])
                    std = torch.tensor(self.train_stats['std_energy'])
                    feat = torch.tensor(feat, dtype=torch.float32)
                    feat = (feat - mean) / (std + 1e-8)

                elif mod == 'oversmooth':
                    feat = extract_proxy_scalars_LSF(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                    feat = torch.tensor(feat, dtype=torch.float32)
                    mean = torch.tensor(self.train_stats['mean_smooth'])
                    std = torch.tensor(self.train_stats['std_smooth'])
                    feat = (feat - mean) / (std + 1e-8)

                else:
                    feat = torch.tensor(feat, dtype=torch.float32)

                feat_dict[mod] = feat

        return feat_dict, label, mod_gt

class LoadEvalData_MoE_SanityCheck(Dataset):
    def __init__(self, list_IDs, models_list, config):
        self.list_IDs = list_IDs
        self.fs = config['fs']
        self.win_len = config['win_len']
        self.stft_params = STFTWindowParams()

        self.models_list = models_list
        self.config = config

        self.selected_mods = config['selected']
        self.gt_f0 = os.path.join(config['rootpath'], "temp_feat", f"f0_{config['test_data']}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        path = self.list_IDs[idx]

        if self.config['train_data'] == 'asv19':
            self.train_stats = self.config["train_stats_asv19"]
        elif self.config['train_data'] == 'asv5':
            self.train_stats = self.config["train_stats_asv5"]
        else:
            print(f"Not recognized: {self.config['train_data']}. No available normalization data.")

        mod_list = []
        label = 0
        track_name = os.path.splitext(os.path.basename(path))[0]

        if len(self.selected_mods) > 1 or len(self.models_list) > 1:
            registry = MANIPULATIONS_REGISTRY.copy()
            for model_type in self.models_list:
                manipulation = None
                for m, mod in registry:
                    if m == model_type:
                        manipulation = mod
                        registry.remove((m, mod))
                        break
                if manipulation is None:
                    raise ValueError(f"No manipulation found for model {model_type}")
                mod_list.append(manipulation)
        else:
            mod_list.append(self.selected_mods[0])

        # EXTRACT FEATURES
        x = load_audio_track(path, trim=False, normalize=True)
        x = torch.Tensor(x)
        feat_dict = {}
        invalid_mods = set()
        for mod in mod_list:
            if mod in ['flat_pitch', 'f0_incons']:
                file = os.path.join(self.gt_f0, f'{track_name}.npy')

                if not os.path.exists(file):
                    x = load_audio_track(path, trim=True, normalize=True)  # trim=True !
                    feat = extract_feature(x, mod, self.stft_params)
                    np.save(file, feat)
                else:
                    feat = np.load(file)
                    feat = np.nan_to_num(feat, nan=0.0)

                if np.all(feat == 0.0):
                    label = -1
                    feat = np.zeros(10)
                    feat = torch.tensor(feat, dtype=torch.float32)
                    invalid_mods.add(mod)

            else:
                x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=True)
                feat = extract_feature(x.numpy(), mod, self.stft_params)

            feat_dict[mod] = feat

        # APPLY SELECTED MANIPULATION(S) + EXTRACT PROXY FEATURES WHEN NEEDED

        if idx % 2 == 0 and len(self.selected_mods) > 0:

            for mod in self.selected_mods:
                feat = feat_dict[mod]

                if mod in ['min_phase', 'energy']:
                    feat_mod = apply_manipulation(x.numpy(), mod=mod, stft_params=self.stft_params)
                else:
                    feat_mod = apply_manipulation(feat.numpy() if isinstance(feat, torch.Tensor) else feat, mod=mod, stft_params=self.stft_params)

                feat_dict[mod] = torch.tensor(feat_mod, dtype=torch.float32)

            label = 1

        # Extract proxy features

        for mod in mod_list:
            if mod in invalid_mods:
                feat_dict[mod] = torch.zeros(10, dtype=torch.float32)
                continue

            feat = feat_dict[mod]
            if mod in ['f0_incons', 'flat_pitch']:
                feat_new = extract_proxy_scalars_F0(feat.numpy() if isinstance(feat, torch.Tensor) else feat)

                if mod == 'f0_incons':
                    mean = torch.tensor(self.train_stats['mean_incons'])
                    std = torch.tensor(self.train_stats['std_incons'])
                else:
                    mean = torch.tensor(self.train_stats['mean_flat'])
                    std = torch.tensor(self.train_stats['std_flat'])

                if np.all(feat_new == 0.0):
                    label = -1
                    feat_new = torch.zeros(10, dtype=torch.float32)
                else:
                    feat_new = torch.tensor(feat_new, dtype=torch.float32)
                    feat_new = (feat_new - mean) / (std + 1e-8)

                feat_new = torch.tensor(feat_new, dtype=torch.float32)

            else:
                feat_new = torch.tensor(feat, dtype=torch.float32)

            feat_dict[mod] = feat_new

        return feat_dict, label, track_name

class LoadEvaluationData(Dataset):
    def __init__(self, list_IDs, labels, models_list, config):
        self.list_IDs = list_IDs
        self.labels = labels

        self.fs = config['fs']
        self.win_len = config['win_len']
        self.stft_params = STFTWindowParams()

        self.mod = config['mod']
        self.train_domain = config['train_data']

        self.models_list = models_list
        self.config = config

        self.gt_f0 = os.path.join(config['rootpath'], "temp_feat", f"f0_{config['test_data']}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        path = self.list_IDs[idx]

        if self.train_domain == 'asv19':
            self.train_stats = self.config["train_stats_asv19"]
        elif self.train_domain == 'asv5':
            self.train_stats = self.config["train_stats_asv5"]
        else:
            print(f"Not recognized: {self.train_domain}. No available normalization data.")

        label = self.labels[idx]
        track_name = os.path.splitext(os.path.basename(path))[0]

        mod_list = []
        if self.mod == 'all':  # for MoE
            registry = MANIPULATIONS_REGISTRY.copy()
            for model_type in self.models_list:
                manipulation = None
                for m, mod in registry:
                    if m == model_type:
                        manipulation = mod
                        registry.remove((m, mod))
                        break
                if manipulation is None:
                    raise ValueError(f"No manipulation found for model {model_type}")
                mod_list.append(manipulation)
        else:  # individual exp / ensemble
            mod_list.append(self.mod)

        # EXTRACT FEATURES

        feat_dict = {}
        invalid_mods = set()

        x = load_audio_track(path, trim=False, normalize=True)
        x = torch.Tensor(x)

        for mod in mod_list:

            if mod in ['flat_pitch', 'f0_incons']:
                file = os.path.join(self.gt_f0, f'{track_name}.npy')

                if not os.path.exists(file):
                    x = load_audio_track(path, trim=True, normalize=True)  # trim=True !
                    feat = extract_feature(x, mod, self.stft_params)
                    np.save(file, feat)
                else:
                    feat = np.load(file)
                    feat = np.nan_to_num(feat, nan=0.0)

                if np.all(feat == 0.0):
                    label = -1
                    feat = np.zeros(10)
                    feat = torch.tensor(feat, dtype=torch.float32)
                    invalid_mods.add(mod)

            else:
                x = prepare_audio_window(x, (self.win_len * self.fs), is_eval=True)
                feat = extract_feature(x, mod, self.stft_params)

            feat_dict[mod] = feat

        # EXTRACT PROXY FEATURES WHEN NEEDED

        for mod in mod_list:

            if mod in invalid_mods:
                feat_dict[mod] = torch.zeros(10, dtype=torch.float32)
                continue

            feat = feat_dict[mod]

            if mod in ['f0_incons', 'flat_pitch']:

                feat = extract_proxy_scalars_F0(feat.numpy() if isinstance(feat, torch.Tensor) else feat)

                if mod == 'f0_incons':
                    mean = torch.tensor(self.train_stats['mean_incons'])
                    std = torch.tensor(self.train_stats['std_incons'])
                else:
                    mean = torch.tensor(self.train_stats['mean_flat'])
                    std = torch.tensor(self.train_stats['std_flat'])

                if np.all(feat == 0.0):
                    label = -1
                    feat = torch.zeros(10, dtype=torch.float32)
                else:
                    feat = torch.tensor(feat, dtype=torch.float32)
                    feat = (feat - mean) / (std + 1e-8)

                feat = torch.tensor(feat, dtype=torch.float32)

            elif mod == 'energy':
                feat = extract_proxy_scalars_energy(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                mean = torch.tensor(self.train_stats['mean_energy'])
                std = torch.tensor(self.train_stats['std_energy'])
                feat = torch.tensor(feat, dtype=torch.float32)
                feat = (feat - mean) / (std + 1e-8)

            elif mod == 'oversmooth':
                feat = extract_proxy_scalars_LSF(feat.numpy() if isinstance(feat, torch.Tensor) else feat)
                feat = torch.tensor(feat, dtype=torch.float32)
                mean = torch.tensor(self.train_stats['mean_smooth'])
                std = torch.tensor(self.train_stats['std_smooth'])
                feat = (feat - mean) / (std + 1e-8)

            else:
                feat = torch.tensor(feat, dtype=torch.float32)

            feat_dict[mod] = feat

        return feat_dict, label, track_name

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
        y = self.labels[index]

        x = load_audio_track(track, sr=self.fs)
        x = prepare_audio_window(x, self.win_len_samples, is_eval=True)

        return x, y, track
