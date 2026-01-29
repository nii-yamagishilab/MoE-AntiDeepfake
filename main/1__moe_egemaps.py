import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.utils import *
from sklearn.linear_model import LogisticRegression
import pandas as pd
import opensmile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# class MOE_eGeMAPS(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#
#         return output

def load_asv5(paths, base_path):
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

        df_out = pd.DataFrame({
            "path": df[1].apply(
                lambda x: os.path.join(base_path, subdir, f"{x}.flac")
            ),
            "label": (df[8] == "spoof").astype(int),
            "generator": df[7]
        })

        dfs.append(df_out)

    # return pd.concat(dfs, ignore_index=True)
    df_all = pd.concat(dfs, ignore_index=True)

    # --- bonafide ---
    df_real = (
        df_all[df_all["label"] == 0]
        .sample(n=5000, random_state=config['seed'])
    )

    # --- spoof (stratified by generator) ---
    df_spoof = df_all[df_all["label"] == 1]

    gen_frac = df_spoof["generator"].value_counts(normalize=True)
    n_per_gen = (gen_frac * 5000).round().astype(int)

    spoof_parts = []
    for gen, n in n_per_gen.items():
        df_g = df_spoof[df_spoof["generator"] == gen]
        spoof_parts.append(
            df_g.sample(n=min(n, len(df_g)), random_state=config['seed'])
        )

    df_spoof_sub = pd.concat(spoof_parts)

    df_final = (
        pd.concat([df_real, df_spoof_sub])
        .sample(frac=1, random_state=config['seed'])
        .drop(columns="generator")
        .reset_index(drop=True)
    )

    return df_final

def extract_features(paths, feature_names):
    rows = []

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    for path in tqdm(paths, desc="Extracting eGeMAPS"):
        try:
            y = smile.process_file(path)
            rows.append(y[feature_names].iloc[0].values)
        except Exception as e:
            rows.append([np.nan] * len(feature_names))
            print(f"\nFailed on {path}: {e}")

    return pd.DataFrame(rows, columns=feature_names)

def main():
    Path(config['save_model_folder']).mkdir(parents=True, exist_ok=True)

    df = load_asv5(['/Database/ASVspoof5/protocols/ASVspoof5.train.tsv',
                    '/Database/ASVspoof5/protocols/ASVspoof5.dev.track_1.tsv'],
                   '/Database/ASVspoof5')

    features = [
        "loudness_sma3_stddevFallingSlope",  # F19
        "spectralFluxV_sma3nz_amean",  # F66
        "spectralFlux_sma3_stddevNorm",  # F21
        "MeanUnvoicedSegmentLength",  # F85
        "StddevUnvoicedSegmentLength",  # F86
    ]

    X_train = extract_features(df["path"], features)
    y_train = df["label"].values

    # Experts training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    experts = [LogisticRegression() for _ in range(X_train.shape[1])]

    for i, expert in enumerate(experts):
        expert.fit(X_train[:, i].reshape(-1, 1), y_train)

    # Moe training



if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])
    config['save_model_folder'] = '/home/smg/v-viola/moe_antideepfake/checkpoints/POC'

    main()