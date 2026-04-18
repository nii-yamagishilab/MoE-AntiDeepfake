import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
sys.path.insert(1, os.getcwd())

from pathlib import Path
from utils.data_utils import load_evaluation_data
from models.small_models import *
from models.lcnn_model import LCNN
from utils.training_utils import eval_expert
from utils.utils import *
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_eval(model_names, experts, checkpoint_paths, config):
    Path(config['save_path']).mkdir(parents=True, exist_ok=True)
    print(f"Result path: {config['save_path']}")

    eval_loader = load_evaluation_data(model_names, config)

    for expert, model_path in zip(experts, checkpoint_paths):
        with torch.no_grad():
            file_IDs, scores, labels = eval_expert(expert, eval_loader, device)

        scores = torch.tensor(scores)
        scores = scores.cpu().numpy()
        labels = np.array(labels)
        print(f"Valid samples: {len(labels)}")

        rocauc, eer = compute_metrics_v2(labels, scores)
        print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n")

        np.save(f"{config['save_path']}/file_IDs.npy", file_IDs)
        np.save(f"{config['save_path']}/scores.npy", scores)
        np.save(f"{config['save_path']}/labels.npy", labels)


def main(checkpoint_paths, model_classes):

    # DEFINE MODEL
    experts = []
    for model_path, ModelClass in zip(checkpoint_paths, model_classes):
        expert = ModelClass(return_emb=False)
        expert = expert.to(device)
        expert.load_state_dict(torch.load(model_path, map_location=device))
        experts.append(expert)
        print(f'Model loaded: {os.path.basename(model_path)}\n')

    # LOAD EVAL DATA
    model_names = []
    for i in range(len(checkpoint_paths)):
        name = Path(checkpoint_paths[i]).stem.rsplit('_', 1)[-1]
        model_names.append(name)

    if config['eval_all']:
        if config['target_check']:
            config['save_path'] = os.path.join(config['save_folder'], 'ALL')
        else:
            config['save_path'] = config['save_folder']

        run_eval(model_names, experts, checkpoint_paths, config)

    else:
        targets = config['selected']
        for target in targets:
            config['selected'] = [target]

            if config['target_check']:
                config['save_path'] = os.path.join(config['save_folder'], target)
            else:
                config['save_path'] = config['save_folder']

            run_eval(model_names, experts, checkpoint_paths, config)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")

    config.update(config_paths)
    del config_paths
    seed_everything(config['seed'])

    # --------------------- PLEASE DEFINE ----------------------#

    config['mod'] = 'ckrboard'
    config['sanity_check'] = False
    config['target_check'] = True

    config['train_data'] = 'mix'
    config['test_data'] = 'asv5'
    config['eval_all'] = False

    config['model'] = 'MLP'
    model_classes = [MLP_Checkerboard]

    model_names = [
        # f'{config['train_data'].upper()}_{config['mod'].upper()}_{config['model'].upper()}.pth',
        # f'{config['train_data'].upper()}_F0_JOINT_{config['model'].upper()}.pth',
        f'{config['train_data'].upper()}_FILTERING_{config['model'].upper()}.pth',
    ]

    # ------------------------- Thanks -------------------------#

    if config['target_check']:
        if config['test_data'] == 'asv19':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
        elif config['test_data'] == 'asv5':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08',
                                  'A09', 'A10', 'A11', 'A12', 'A14',
                                  'A17', 'A19', 'A21', 'A22', 'A28', 'A29']
        elif config['test_data'] == 'wf':
            config['selected'] = ["tts_ljspeech", "ljspeech_full_band_melgan", "ljspeech_hifiGAN", "ljspeech_melgan",
                                  "ljspeech_melgan_large", "ljspeech_multi_band_melgan", "ljspeech_parallel_wavegan",
                                  "ljspeech_waveglow"]
        elif config['test_data'] == 'sc':
            config['selected'] = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11',
                                  'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23']
        elif config['test_data'] == 'asv15':
            config['selected'] = ['S3', 'S4', 'S10']
        elif config['test_data'] == 'mlaad':
            config['selected'] = ['Mars5', 'MatchaTTS', 'MeloTTS', 'Metavoice-1B', 'OpenVoiceV2', 'WhisperSpeech', 'e2-tts',
                                  'f5-tts', 'facebook_mms-tts-eng', 'griffin_lim', 'microsoft_speecht5_tts', 'optispeech',
                                  'parler_tts_large_v1', 'parler_tts_mini_v0.1', 'parler_tts_mini_v1', 'suno_bark-small',
                                  'suno_bark', 'tts_models_en_blizzard2013_capacitron-t2-c50', 'tts_models_en_ek1_tacotron2',
                                  'tts_models_en_jenny_jenny', 'tts_models_en_ljspeech_fast_pitch', 'tts_models_en_ljspeech_glow-tts',
                                  'tts_models_en_ljspeech_neural_hmm', 'tts_models_en_ljspeech_overflow', 'tts_models_en_ljspeech_speedy-speech',
                                  'tts_models_en_ljspeech_tacotron2-DCA', 'tts_models_en_ljspeech_tacotron2-DDC', 'tts_models_en_ljspeech_tacotron2-DDC_ph',
                                  'tts_models_en_ljspeech_vits--neon', 'tts_models_en_ljspeech_vits', 'tts_models_en_multi-dataset_tortoise-v2',
                                  'tts_models_en_sam_tacotron-DDC', 'tts_models_multilingual_multi-dataset_bark', 'tts_models_multilingual_multi-dataset_xtts_v1.1',
                                  'tts_models_multilingual_multi-dataset_xtts_v2', 'vixTTS']
        else:
            print("Test dataset not supported. \n")
            sys.exit()
    else:
        config['selected'] = [config['mod']]

    print("\nEVALUATION MODE\n")
    print(f"Manipulation type: {config['mod']}")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")

    checkpoint_paths = [os.path.join(config['save_model_folder'], 'experts', name) for name in model_names]
    config['save_folder'] = os.path.join(
        config['save_results_folder'],
        'experts',
        config['test_data'],
        f"experts_trained_on_{config['train_data']}",
        config['mod']
    )

    if config['target_check']:
        config['save_folder'] = os.path.join(config['save_folder'], 'TargetCheck')
    else:
        config['save_folder'] = os.path.join(config['save_folder'], 'SanityCheck')

    main(checkpoint_paths, model_classes)



