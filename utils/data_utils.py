import os
import librosa
from glob import glob
import torchaudio
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch
import random
from sklearn.model_selection import train_test_split


############################## Build Dataframes #################################

def load_asvspoof_data(file_name, base_path, label_dir):
    df = pd.read_csv(os.path.join(label_dir, file_name), sep=' ', header=None)
    df = df.rename(columns={0: 'speaker'})
    df['label'] = df[4].map({'bonafide': 0, 'spoof': 1})
    df['path'] = df[1].apply(lambda x: os.path.join(base_path, x + '.flac'))
    return df[['path', 'label', 'speaker']]

def load_for_data(base_dir, subset):
    real_files = glob(os.path.join(base_dir, subset, 'real', '*.wav'))
    fake_files = glob(os.path.join(base_dir, subset, 'fake', '*.wav'))

    df_real = pd.DataFrame(real_files, columns=['path'])
    df_real['label'] = 0

    df_fake = pd.DataFrame(fake_files, columns=['path'])
    df_fake['label'] = 1

    return pd.concat([df_real, df_fake])

def load_training_data_holdout(dataset_name, config):

    if dataset_name == 'asv19':
        label_dir = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_cm_protocols'
        base_paths = {
            'train': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_train/flac/',
            'dev': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_dev/flac/',
        }

        df_train_asv = load_asvspoof_data('ASVspoof2019.LA.cm.train.trn.txt', base_paths['train'], label_dir)
        df_dev_asv = load_asvspoof_data('ASVspoof2019.LA.cm.dev.trl.txt', base_paths['dev'], label_dir)

        df_all = pd.concat([df_train_asv, df_dev_asv], ignore_index=True)
        df_all = df_all[['path', 'label', 'speaker']]

        unique_speakers = df_all['speaker'].unique()
        train_speakers, dev_speakers = train_test_split(unique_speakers, test_size=0.25, random_state=config['seed'],
            stratify=df_all.groupby('speaker')['label'].first())

        df_train = df_all[df_all['speaker'].isin(train_speakers)]
        df_dev = df_all[df_all['speaker'].isin(dev_speakers)]

    elif dataset_name == 'for':
        df_train = load_for_data(config['for_path'], 'training')
        df_dev = load_for_data(config['for_path'], 'validation')

    elif dataset_name == 'itw':

        df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
        df_meta['path'] = df_meta['file'].apply(
            lambda x: os.path.join(config["InTheWild_path"], config["InTheWild_wav_folder"],x))
        df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
        df_meta = df_meta[['path', 'label']]

        df_train, df_temp = train_test_split(
            df_meta,
            test_size=0.3,
            stratify=df_meta['label'],
            random_state=config['seed'])

        df_dev, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=config['seed'])

        Path("csv_files").mkdir(exist_ok=True)
        output_files = {
            'csv_files/itw_train.csv': df_train,
            'csv_files/itw_dev.csv': df_dev,
            'csv_files/itw_test.csv': df_test
        }
        for filename, df in output_files.items():
            if not os.path.exists(filename):
                df.to_csv(filename, index=False)
                print(f"Saved {filename}\n")
            else:
                print(f"{filename} already exists - skipping overwrite")

    elif dataset_name == 'mlaad':

        # ENGLISH ONLY DATA
        fake_data = glob(os.path.join(config['MLAAD_path'], 'en/*/*.wav'))
        fake_labels = [1] * len(fake_data)

        real_data = glob(os.path.join(config['MAILABS_path'], '*/*/*/*/*/*/*.wav'))
        real_data = random.sample(real_data, 30000)
        real_labels = [0] * len(real_data)

        data_all = real_data + fake_data
        labels_all = real_labels + fake_labels

        df_all = pd.DataFrame({'path': data_all, 'label': labels_all})

        df_train, df_temp = train_test_split(
            df_all,
            test_size=0.3,
            stratify=df_all['label'],
            random_state=config['seed'])

        df_dev, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=config['seed'])

        Path("csv_files").mkdir(exist_ok=True)
        output_files = {
            'csv_files/mlaad_en_train.csv': df_train,
            'csv_files/mlaad_en_dev.csv': df_dev,
            'csv_files/mlaad_en_test.csv': df_test
        }

        for filename, df in output_files.items():
            if not os.path.exists(filename):
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")
            else:
                print(f"{filename} already exists - skipping overwrite")

    else:
        raise ValueError(f'Dataset {dataset_name} not recognized')

    return df_train, df_dev

def load_evaluation_data(datasets, config):

    if isinstance(datasets, str):
        datasets = [datasets]

    datasets = {d.lower() for d in datasets}
    data = {}

    # ASVSPOOF 2019
    if 'asv19' in datasets:
        label_dir = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_cm_protocols'
        base_eval = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_eval/flac/'

        df_eval_asv = load_asvspoof_data(
            'ASVspoof2019.LA.cm.eval.trl.txt',
            base_eval,
            label_dir
        )
        data['asv19'] = df_eval_asv

    # FAKE OR REAL
    if 'for' in datasets:
        df_eval_for = load_for_data(config["for_path"], 'testing')
        data['for'] = df_eval_for

    # ADD 2022
    if 'add22' in datasets:
        df_eval_add = pd.read_csv(
            f'{config["add2022_label_path"]}/add2022_track1_label.txt',
            sep=' ',
            header=None
        )
        df_eval_add['path'] = (
            f'{config["add2022_path"]}/track1test/' + df_eval_add[0].astype(str)
        )
        df_eval_add['label'] = df_eval_add[1].map({'genuine': 0, 'fake': 1})
        df_eval_add = df_eval_add[['path', 'label']]

        data['add22'] = df_eval_add

    # IN THE WILD
    if 'itw' in datasets:
        # df_eval = pd.read_csv('csv_files/itw_test.csv')  # use if available
        df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
        df_meta['path'] = df_meta['file'].apply(
            lambda x: os.path.join(config["InTheWild_path"], config["InTheWild_wav_folder"],x))
        df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
        df_meta = df_meta[['path', 'label']]

        _, df_temp = train_test_split(
            df_meta,
            test_size=0.3,
            stratify=df_meta['label'],
            random_state=config['seed']
        )
        _, df_eval = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=config['seed']
        )

        data['itw'] = df_eval

    # MLAAD
    if 'mlaad' in datasets:
        # ENGLISH ONLY DATA
        # df_eval = pd.read_csv('csv_files/mlaad_test.csv')  # use if available
        fake_data = glob(os.path.join(config['MLAAD_path'], 'en/*/*.wav'))
        fake_labels = [1] * len(fake_data)

        real_data = glob(os.path.join(config['MAILABS_path'], '*/*/*/*/*/*/*.wav'))
        real_data = random.sample(real_data, 30000)
        real_labels = [0] * len(real_data)

        data_all = real_data + fake_data
        labels_all = real_labels + fake_labels

        df_all = pd.DataFrame({'path': data_all, 'label': labels_all})

        _, df_temp = train_test_split(
            df_all,
            test_size=0.3,
            stratify=df_all['label'],
            random_state=config['seed'])

        _, df_eval = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=config['seed'])

        data['mlaad'] = df_eval

    # TIMIT-TTS
    if 'timit_tts' in datasets:
        df_timit_fake = pd.DataFrame(
            glob(f'{config["timit_tts_path"]}/CLEAN/single_speaker/*/*.wav') +
            glob(f'{config["timit_tts_path"]}/CLEAN/multi_speaker/*/*/*.wav'),
            columns=['path']
        )
        df_timit_fake['label'] = 1

        df_timit_real = pd.DataFrame(
            glob(f'{config["vidtimit_path"]}/*/audio/*.wav'),
            columns=['path']
        )
        df_timit_real['label'] = 0

        df_timit = pd.concat([df_timit_fake, df_timit_real], ignore_index=True)
        data['timit_tts'] = df_timit

    # ASVSPOOF 2021 DF
    if 'asv21' in datasets:
        df_asv21 = pd.read_csv(
            config["asvspoof21_df_meta"],
            sep=' ',
            header=None
        )

        df_asv21 = df_asv21[df_asv21[7] == 'eval']
        df_asv21 = df_asv21[df_asv21[3] != 'asvspoof']
        df_asv21 = df_asv21[df_asv21[2] == 'nocodec']

        df_asv21['path'] = df_asv21[1].apply(
            lambda x: os.path.join(
                config["asvspoof21_df_path"], x + '.flac'
            )
        )
        df_asv21['label'] = df_asv21[5].map({'bonafide': 0, 'spoof': 1})
        df_asv21 = df_asv21[['path', 'label']].reset_index(drop=True)

        data['asv21'] = df_asv21

    # ASVSPOOF 5
    if 'asv5' in datasets:
        df_asv5 = pd.read_csv(
            config["asvspoof5_meta"],
            sep=' ',
            header=None
        )

        df_asv5 = df_asv5[df_asv5[3] == '-']  # no codec
        df_asv5 = df_asv5.groupby(7, group_keys=False).apply(
            lambda x: x.sample(n=7760, random_state=config['seed'])
        )

        df_asv5['label'] = df_asv5[8].map({'bonafide': 0, 'spoof': 1})
        df_asv5['path'] = df_asv5[1].apply(
            lambda x: os.path.join(config["asvspoof5_path"], x + '.flac')
        )

        df_asv5 = df_asv5[['path', 'label']].reset_index(drop=True)
        data['asv5'] = df_asv5

    return data

def load_asv5_bonafide(paths, base_path):
    dfs = []

    for p in paths:
        filename = os.path.basename(p).lower()
        if "train" in filename:
            subdir = "flac_T"
        elif "dev" in filename:
            subdir = "flac_D"
        else:
            subdir = "flac_E_eval"

        df = pd.read_csv(p, sep=' ', header=None)
        # df = df[df[2] == 'F']
        df = (
            df[df[8] == 'bonafide']
            .assign(path=lambda df: df[1].apply(
                lambda x: os.path.join(base_path, subdir, f"{x}.flac")
            ))[['path']]
        )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

############################## Feed Dataloader ##################################

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

class LoadTrainData(Dataset):
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

class LoadEvalData(Dataset):
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

class LoadExperimentalData(Dataset):
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

        if not self.is_eval:
            if random.random() < self.p_perturb:
                if self.multi_band:
                    band = random.choice(self.bands)
                    mag = self.add_band_limited_noise(mag, band)
                else:
                    mag = self.add_band_limited_noise(mag, (self.f_low, self.f_high))

                label = 1
            mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
            return mag, label

        else:
            mag = self.add_band_limited_noise(mag, (self.f_low, self.f_high))
            mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
            return mag, label, key

        # # Shortcut 4 real data, comment if-else above
        # mag = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0)
        # return mag, label, key