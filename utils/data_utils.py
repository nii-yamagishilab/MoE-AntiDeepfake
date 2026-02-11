import os
import sys
from glob import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.make_artifacts import *
from utils.dataloader import LoadTrainData, LoadTrainData_v2
from torch.utils.data import DataLoader


#--------------------------- Current Dataframes ---------------------------#

def load_training_data(models_list, config):

    if not config['merge']:
        df_train, df_dev = make_df_bonafide(config)
    else:
        df_all = make_df_bonafide(config)
        df_train, df_dev = train_test_split(
            df_all,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

    df_train = pd.DataFrame(df_train.values.repeat(2, axis=0), columns=df_train.columns)
    df_dev = pd.DataFrame(df_dev.values.repeat(2, axis=0), columns=df_dev.columns)

    # Define training dataloader
    file_train = list(df_train['path'])
    train_set = LoadTrainData(list_IDs=file_train, config=config)
    # train_set = LoadTrainData_v2(list_IDs=file_train, models_list=models_list, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    del train_set

    # Define validation dataloader
    file_dev = list(df_dev['path'])
    dev_set = LoadTrainData(list_IDs=file_dev, config=config)
    # dev_set = LoadTrainData_v2(list_IDs=file_dev, models_list=models_list, config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False)
    del dev_set

    print(f'Total no. of training samples: {len(train_loader.dataset)}')
    print(f'Total no. of dev samples: {len(dev_loader.dataset)} \n')

    return train_loader, dev_loader

def make_df_bonafide(config):
    if config['train_data'] not in config:
        print('Dataset not supported.')
        sys.exit()

    ds_name = config['train_data']
    ds = config[ds_name]
    base_path = ds['base_path']

    def load_subset(file_path, dir_name, filter_col, path_col):
        df = pd.read_csv(file_path, sep=' ', header=None)
        return (
            df[df[filter_col] == 'bonafide']
              .assign(path=lambda df: df[path_col].apply(
                  lambda x: os.path.join(base_path, dir_name, f"{x}.flac")
              ))[['path']]
        )

    df_t = load_subset(os.path.join(base_path, ds['train_file']),
                       ds['train_dir'],
                       ds['filter_column'],
                       ds['path_column'])

    df_d = load_subset(os.path.join(base_path, ds['dev_file']),
                       ds['dev_dir'],
                       ds['filter_column'],
                       ds['path_column'])

    if config['merge']:
        df = pd.concat([df_t, df_d], ignore_index=True)
        return df
    else:
        return df_t, df_d


#--------------------------- Baseline Dataframes ---------------------------#

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