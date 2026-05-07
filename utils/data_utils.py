import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.dataloader import *
from torch.utils.data import DataLoader
from utils.model_utils import *
from glob import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#--------------------------- Current Dataframes ---------------------------#

def make_df_bonafide(config):
    def load_subset(file_path, dir_name, filter_col, path_col):
        df = pd.read_csv(file_path, sep=' ', header=None)
        return (
            df[df[filter_col] == 'bonafide']
            .rename(columns={0: 'speaker'})
            .assign(path=lambda df: df[path_col].apply(
                lambda x: os.path.join(base_path, dir_name, f"{x}.flac")
            ))[["speaker", "path"]]
        )

    if config['large_scale']:

        # def speaker_split(df, val_ratio=0.2, seed=42):
        #     speakers = df['speaker'].drop_duplicates().sample(frac=1, random_state=seed)
        #     n_val = int(len(speakers) * val_ratio)
        #
        #     val_speakers = set(speakers.iloc[:n_val])
        #     train_speakers = set(speakers.iloc[n_val:])
        #
        #     df_train = df[df['speaker'].isin(train_speakers)]
        #     df_val = df[df['speaker'].isin(val_speakers)]
        #
        #     return df_train, df_val
        #
        # # Load ASVspoof 2019 & 5
        # dfs = {}
        # for ds_name in ['asv19', 'asv5']:
        #     ds = config[ds_name]
        #     base_path = ds['base_path']
        #     dfs[ds_name] = pd.concat([
        #         load_subset(os.path.join(base_path, ds[f'{split}_file']),
        #                     ds[f'{split}_dir'],
        #                     ds['filter_column'],
        #                     ds['path_column'])
        #         for split in ['train', 'dev']
        #     ], ignore_index=True)
        # df_asv19 = dfs['asv19']
        # df_asv5 = dfs['asv5']
        #
        # # Load VoxCeleb 1
        # ds_name = 'sc'
        # ds = config[ds_name]
        # df_t = pd.read_csv(ds['real_T_dir'], header=None, names=['path'])
        # df_d = pd.read_csv(ds['real_D_dir'], header=None, names=['path'])
        #
        # df_vox = pd.concat([df_t, df_d], ignore_index=True)
        # df_vox['speaker'] = df_vox['path'].str.split('/').str[-2]
        #
        # n = min(len(df_asv19), len(df_asv5), len(df_vox))
        # df_asv19 = df_asv19.sample(n, random_state=42)
        # df_asv5 = df_asv5.sample(n, random_state=42)
        # df_vox = df_vox.sample(n, random_state=42)
        #
        # asv19_tr, asv19_val = speaker_split(df_asv19)
        # asv5_tr, asv5_val = speaker_split(df_asv5)
        # vox_tr, vox_val = speaker_split(df_vox)
        #
        # df_t = pd.concat([asv19_tr, asv5_tr, vox_tr], ignore_index=True)
        # df_d = pd.concat([asv19_val, asv5_val, vox_val], ignore_index=True)
        #
        # df_t['path'].to_csv('/home/smg/v-viola/Mix_Train.txt',
        #                      index=False, header=False)
        #
        # df_d['path'].to_csv('/home/smg/v-viola/Mix_Dev.txt',
        #                    index=False, header=False)

        df_t = pd.read_csv('/home/smg/v-viola/Mix_Train.txt',
                            header=None, names=['path'])

        df_d = pd.read_csv('/home/smg/v-viola/Mix_Dev.txt',
                          header=None, names=['path'])

        return df_t, df_d

    else:
        ds_name = config['train_data']
        ds = config[ds_name]
        base_path = ds['base_path']

        df_t = load_subset(os.path.join(base_path, ds['train_file']),
                           ds['train_dir'],
                           ds['filter_column'],
                           ds['path_column'])

        df_d = load_subset(os.path.join(base_path, ds['dev_file']),
                           ds['dev_dir'],
                           ds['filter_column'],
                           ds['path_column'])

        return pd.concat([df_t, df_d], ignore_index=True)

def load_training_data(models_list, config):
    if config['large_scale']:
        df_train, df_dev = make_df_bonafide(config)
    else:
        df_all = make_df_bonafide(config)
        df_train, df_dev = train_test_split(
            df_all,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

    if config['reduce_set']:
        df_train = df_train.sample(frac=0.15, random_state=42).reset_index(drop=True)
        df_dev = df_dev.sample(frac=0.15, random_state=42).reset_index(drop=True)

    df_train = pd.DataFrame(df_train.values.repeat(2, axis=0), columns=df_train.columns)
    df_dev = pd.DataFrame(df_dev.values.repeat(2, axis=0), columns=df_dev.columns)

    # Define training dataloader
    file_train = list(df_train['path'])
    train_set = LoadTrainData(list_IDs=file_train, models_list=models_list, config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    del train_set

    # Define validation dataloader
    file_dev = list(df_dev['path'])
    # file_dev = random.sample(file_dev, 3000)
    dev_set = LoadTrainData(list_IDs=file_dev, models_list=models_list, config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False)
    del dev_set

    print(f'Total no. of training samples: {len(train_loader.dataset)}')
    print(f'Total no. of dev samples: {len(dev_loader.dataset)} \n')

    return train_loader, dev_loader

def load_calibration_data(mod, config):
    print("Loading calibration data.")
    nb_class_samples = config['calib_class_samples']

    # Load real data from the validation set (~3000 samples max)
    if config['train_data'] == 'mix':
        config['large_scale'] = True
        _, df_real = make_df_bonafide(config)
    else:
        df_all = make_df_bonafide(config)
        _, df_real = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True)
    df_real['label'] = 0

    # Load an equal number of fake samples
    if config['calibration_data'] == 'pseudo':
        df_real = df_real.sample(n=nb_class_samples, random_state=42)

        df_eval = pd.DataFrame(df_real.values.repeat(2, axis=0), columns=df_real.columns)
        file_eval = list(df_eval['path'])
        models_list = None

        eval_set = LoadEvalData_MoE_SanityCheck(list_IDs=file_eval, models_list=models_list, config=config)
        print('Using pseudo-fake data.')
    else:
        ds = config[config['calibration_data']]

        if config['calibration_data'] == 'asv15':
            if not config['pool_calibration_set']:
                if mod in ['flat_pitch', 'f0_incons', 'energy']:
                    target = ['S10']
                else:
                    target = ['S3']
            else:
                target = ['S3', 'S10']

            print(target)
            nb_targets = len(target)

            df_test = pd.read_csv(os.path.join(ds['base_path'], "CM_protocol/cm_evaluation.ndx"), sep=' ',
                                  header=None)
            df_test = df_test[df_test[ds['gen_column']].isin(target)]
            df_test['label'] = (df_test[ds['gen_column']] != 'human').astype(int)
            df_test['path'] = (
                        ds['file_dir'] + '/' + df_test[0].astype(str) + '/' + df_test[1].astype(str) + '.wav')

            dfs = []

            df_real = df_real.sample(n=nb_class_samples * nb_targets, random_state=42)
            dfs.append(df_real)

            df_fake = (
                df_test[df_test['label'] == 1]
                .groupby(ds['gen_column'], group_keys=False)
                .sample(n=nb_class_samples, random_state=42))
            dfs.append(df_fake)

            df_eval = pd.concat(dfs, ignore_index=True)

        elif config['calibration_data'] in ['asv19', 'asv5']:
            if config['calibration_data'] == 'asv19':
                if not config['pool_calibration_set']:
                    if mod in ['oversmooth', 'tonal', 'filtering']:
                        # target = ['-', 'A01']
                        target = ['A01']
                    elif mod in ['f0_incons', 'energy']:
                        target = ['A04']
                    else:  # phase
                        target = ['A03']
                else:
                    target = ['A01', 'A03', 'A04']
            else:
                if not config['pool_calibration_set']:
                    print("Scenario not defined.\n")
                    sys.exit()
                else:
                    target = ['A02', 'A04', 'A07', 'A08']  # all unique TTS config from training set

            print(target)
            nb_targets = len(target)

            df_t = pd.read_csv(os.path.join(ds['base_path'], ds['train_file']), sep=' ', header=None)
            df_t = df_t[df_t[ds['gen_column']].isin(target)]
            df_t['label'] = df_t[ds['filter_column']].map({'bonafide': 0, 'spoof': 1})
            df_t['path'] = df_t[ds['path_column']].apply(
                lambda x: os.path.join(ds['base_path'], ds['train_dir'], f"{x}.flac"))

            df = pd.concat([df_t], ignore_index=True)
            dfs = []

            df_real = df_real.sample(n=nb_class_samples * nb_targets, random_state=42)
            dfs.append(df_real)

            df_fake = (
                df[df['label'] == 1]
                .groupby(ds['gen_column'], group_keys=False)
                .sample(n=nb_class_samples, random_state=42))
            dfs.append(df_fake)

            df_eval = pd.concat(dfs, ignore_index=True)

        elif config['calibration_data'] == 'sc':
            if not config['pool_calibration_set']:
                print("Scenario not defined.\n")
                sys.exit()
            else:
                target = ['a01', 'a02', 'a03', 'a04', 'a05', 'a08', 'a09', 'a10']

            print(target)
            nb_targets = len(target)
            dfs = []

            df_real = df_real.sample(n=nb_class_samples * nb_targets, random_state=42)
            dfs.append(df_real)

            df_sc = pd.read_csv(os.path.join(ds['base_path'], ds['meta_train']), sep=',')
            df_sc['label'] = df_sc[ds['gen_column']].apply(lambda x: 0 if x == 'a00' else 1)
            df_sc['path'] = df_sc[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['train_dir'], f"{x}"))

            df_fake = (
                df_sc[df_sc['label'] == 1]
                .groupby(ds['gen_column'], group_keys=False)
                .sample(n=nb_class_samples, random_state=42))
            dfs.append(df_fake)

            df_eval = pd.concat(dfs, ignore_index=True)

        else:
            print("Calibration data not supported.")
            sys.exit()

        file_eval = list(df_eval['path'])
        labels_eval = list(df_eval['label'])
        attack_eval = None
        models_list = None
        eval_set = LoadEvaluationData(list_IDs=file_eval, labels=labels_eval, attacks=attack_eval,
                                      models_list=models_list, config=config)

    eval_loader = DataLoader(eval_set, batch_size=config['batch_size_eval'], shuffle=False)
    del eval_set
    print(f'Total no. of samples: {len(eval_loader.dataset)} \n')

    return eval_loader

def load_evaluation_data(models_list, config):
    ds = config[config['test_data']]

    if config['sanity_check']:
        print("Test on manipulated data for sanity checking.")
        df_eval = pd.read_csv(os.path.join(ds['base_path'], ds['train_file']), sep=' ', header=None)

        df_eval = (
            df_eval[df_eval[ds['gen_column']].isin(['-', 'bonafide'])]
            .sample(n=1000, random_state=42)  # subsample for brevity
            .assign(path=lambda df: df[1].apply(
                lambda x: os.path.join(ds['base_path'], ds['train_dir'], f"{x}.flac")
            ))
            [['path']]
        )
        df_eval = pd.DataFrame(df_eval.values.repeat(2, axis=0), columns=df_eval.columns)
        file_eval = list(df_eval['path'])
        eval_set = LoadEvalData_MoE_SanityCheck(list_IDs=file_eval, models_list=models_list, config=config)
    else:
        print("Test on target deepfake data.")

        if config['test_data'] == 'asv15':
            df_test = pd.read_csv(os.path.join(ds['base_path'], "CM_protocol/cm_evaluation.ndx"), sep=' ',header=None)
            df_test = df_test[df_test[ds['gen_column']].isin(['human', 'S3', 'S4', 'S10'])]
            df_test['label'] = (df_test[ds['gen_column']] != 'human').astype(int)
            df_test['path'] = (ds['file_dir'] + '/' + df_test[0].astype(str) + '/' + df_test[1].astype(str) + '.wav')
            target_df = config['selected']
            print(f"Target generator: {target_df}")

            dfs = []

            df_real = df_test[df_test['label'] == 0]
            # n_requested = nb_class_samples * len(target_df)
            # n_available = len(df_real)
            # df_real = df_real.sample(n=min(n_requested, n_available),random_state=42)

            dfs.append(df_real)

            df_fake = (
                df_test[df_test[ds['gen_column']].isin(target_df)]
                # .groupby(ds['gen_column'], group_keys=False).sample(n=nb_class_samples, random_state=42)
                )
            dfs.append(df_fake)

            df_eval = pd.concat(dfs, ignore_index=True)

        elif config['test_data'] == 'wf':
            df_all = pd.read_csv(os.path.join(ds['base_path'], ds['protocols']), sep=' ', header=None)
            df_all['label'] = df_all[ds['filter_column']].map({'bonafide': 0, 'spoof': 1})
            df_all['path'] = df_all[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['file_dir'], f"{x}.wav"))

            df_real = df_all[df_all[ds['gen_column']] == '-']
            df_fake = df_all[df_all[ds['gen_column']] != '-']

            target_df = config['selected']
            print(f"Target generator: {target_df}")

            df_fake = (
                df_fake[df_fake[ds['gen_column']].isin(target_df)]
            )
            df_eval = pd.concat([df_fake, df_real])

        elif config['test_data'] == 'mlaad':
            # Fake samples - english only
            fake_data = glob(os.path.join(ds['fake_path'], 'en/*/*.wav'))
            df_fake = pd.DataFrame({
                "path": fake_data,
                "label": 1,
                "generator": [os.path.basename(os.path.dirname(p)) for p in fake_data]
            })

            target_df = config['selected']
            print(f"Target generator: {target_df}")

            df_fake = (
                df_fake[df_fake['generator'].isin(target_df)]
            )

            # Known real samples
            df_real = pd.read_csv('/Database/ASVspoof2019/public_released_data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', sep=' ', header=None)
            df_real['label'] = df_real[4].map({'bonafide': 0, 'spoof': 1})
            df_real['path'] = df_real[1].apply(lambda x: os.path.join('/Database/ASVspoof2019/public_released_data/LA/ASVspoof2019_LA_eval/flac', f"{x}.flac"))
            df_real.rename(columns={df_real.columns[3]: 'generator'}, inplace=True)
            df_real = df_real[df_real['label'] == 0]

            # # Correct paired real samples
            # real_data = glob('/Database/M-AILABS/*/*/*/*/*/*/*.wav')
            # real_labels = [0] * len(real_data)
            # real_generators = ['-'] * len(real_data)
            # df_real = pd.DataFrame({
            #     "path": real_data,
            #     "label": real_labels,
            #     "generator": real_generators
            # })

            # # DEBUG FILTERING
            # df_fake = df_fake[df_fake['generator'] == 'MatchaTTS']
            # df_fake = df_fake[:100]
            # df_real = df_real[:100]

            df_eval = pd.concat([df_fake, df_real]).reset_index(drop=True)

        elif config['test_data'] == 'sc':
            df_train = pd.read_csv(os.path.join(ds['base_path'], ds['meta_train']), sep=',')
            df_train['label'] = df_train[ds['gen_column']].apply(lambda x: 0 if x == 'a00' else 1)
            df_train['path'] = df_train[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['train_dir'], f"{x}"))

            df_dev = pd.read_csv(os.path.join(ds['base_path'], ds['meta_dev']), sep=',')
            df_dev['label'] = df_dev[ds['gen_column']].apply(lambda x: 0 if x == 'a00' else 1)
            df_dev['path'] = df_dev[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['dev_dir'], f"{x}"))

            df_test = pd.read_csv(os.path.join(ds['base_path'], ds['meta_eval']), sep=',')
            df_test['label'] = df_test[ds['gen_column']].apply(lambda x: 0 if x == 'a00' else 1)
            df_test['path'] = df_test[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['eval_dir'], f"{x}"))

            df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)

            target_df = config['selected']
            print(f"Target generator: {target_df}")

            dfs = []
            df_real = df_test[df_test['label'] == 0]
            dfs.append(df_real)  # ~9k samples

            df_fake = (
                df_all[df_all[ds['gen_column']].isin(target_df)]
                .groupby(ds['gen_column'], group_keys=False).sample(n=7963, random_state=42)
            )
            dfs.append(df_fake) # ~183k samples (if I don't cut: ~2.5 mln!)

            df_eval = pd.concat(dfs, ignore_index=True)

        else:  # ASVspoof 2019 or ASVspoof 5
            if config['test_data'] == 'asv5':
                df_train = pd.read_csv(os.path.join(ds['base_path'], ds['train_file']), sep=' ', header=None)
                df_train['label'] = df_train[ds['filter_column']].map({'bonafide': 0, 'spoof': 1})
                df_train['path'] = df_train[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['train_dir'], f"{x}.flac"))

            df_dev = pd.read_csv(os.path.join(ds['base_path'], ds['dev_file']), sep=' ', header=None)
            df_dev['label'] = df_dev[ds['filter_column']].map({'bonafide': 0, 'spoof': 1})
            df_dev['path'] = df_dev[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['dev_dir'], f"{x}.flac"))

            df_test = pd.read_csv(os.path.join(ds['base_path'], ds['eval_file']), sep=' ', header=None)
            df_test['label'] = df_test[ds['filter_column']].map({'bonafide': 0, 'spoof': 1})
            df_test['path'] = df_test[ds['path_column']].apply(lambda x: os.path.join(ds['base_path'], ds['eval_dir'], f"{x}.flac"))

            if config['test_data'] == 'asv5':
                df_test = df_test[df_test[3] == '-']  # remove post-processed samples
                df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
            else:
                df_all = pd.concat([df_dev, df_test], ignore_index=True)

            target_df = config['selected']
            print(f"Target generator: {target_df}")

            dfs = []
            # nb. we take real samples from the evaluation set only
            df_real = df_test[df_test['label'] == 0]
            dfs.append(df_real)

            if config['test_data'] == 'asv5':
                df_fake = ( df_all[df_all[ds['gen_column']].isin(target_df)]
                            .groupby(ds['gen_column'], group_keys=False).sample(n=7760, random_state=42))
            else:
                df_fake = df_all[df_all[ds['gen_column']].isin(target_df)]

            dfs.append(df_fake)
            df_eval = pd.concat(dfs, ignore_index=True)

        file_eval = list(df_eval['path'])
        labels_eval = list(df_eval['label'])
        attack_eval = list(df_eval[ds['gen_column']])
        eval_set = LoadEvaluationData(list_IDs=file_eval, labels=labels_eval, attacks=attack_eval, models_list=models_list, config=config)

    eval_loader = DataLoader(eval_set, batch_size=config['batch_size_eval'], shuffle=False)
    del eval_set
    print(f'Total no. of samples: {len(eval_loader.dataset)} \n')

    return eval_loader


#--------------------------- Not used - baseline dfs ---------------------------#

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
        train_speakers, dev_speakers = train_test_split(unique_speakers, test_size=0.25, random_state=42,
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
            random_state=42)

        df_dev, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=42)

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
            random_state=42)

        df_dev, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=42)

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

def load_evaluation_data_baseline(datasets, config):

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
            random_state=42
        )
        _, df_eval = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=42
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
            random_state=42)

        _, df_eval = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['label'],
            random_state=42)

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
            lambda x: x.sample(n=7760, random_state=42)
        )

        df_asv5['label'] = df_asv5[8].map({'bonafide': 0, 'spoof': 1})
        df_asv5['path'] = df_asv5[1].apply(
            lambda x: os.path.join(config["asvspoof5_path"], x + '.flac')
        )

        df_asv5 = df_asv5[['path', 'label']].reset_index(drop=True)
        data['asv5'] = df_asv5

    return data