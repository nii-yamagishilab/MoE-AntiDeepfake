import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import os
sys.path.insert(1, os.getcwd())
from pathlib import Path
from models.moe_models import *
from utils.data_utils import load_evaluation_data
from utils.training_utils import eval_moe_compound
from utils.utils import *
from utils.model_utils import MODEL_REGISTRY, get_model_type_from_ckpt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_eval(moe_model, models_list, result_path, config):
    Path(result_path).mkdir(parents=True, exist_ok=True)
    print(f"Result path: {result_path} \n")

    eval_loader = load_evaluation_data(models_list, config)

    with torch.no_grad():
        scores, labels = eval_moe_compound(moe_model, eval_loader, result_path, device)

    scores = np.array(scores)
    labels = np.array(labels)

    print(f"Valid samples: {len(labels)}")
    print(f"Num fake: {(labels == 1).sum()}, Num real: {(labels == 0).sum()}")

    fake_scores = scores[labels == 1]
    real_scores = scores[labels == 0]

    print(f"Mean score (fake=1): {fake_scores.mean():.6f}")
    print(f"Mean score (real=0): {real_scores.mean():.6f}")

    rocauc, eer = compute_metrics_v2(labels, scores)
    print(f"\nAUC: {rocauc * 100:.2f}%, EER: {eer * 100:.2f}%\n\n")


def test_moe(checkpoint_moe, ckpts_experts, save_folder, config):
    checkpoint_name = os.path.basename(os.path.splitext(checkpoint_moe)[0])

    # LOAD EXPERT MODELS
    experts = []
    models_list = []
    model_registry = MODEL_REGISTRY.copy()

    for ckpt in ckpts_experts:
        model_type = get_model_type_from_ckpt(ckpt)
        for name, mod in model_registry:
            if name == model_type:
                model_class = mod
                model_registry.remove((name, model_class))
                break
        model = model_class(return_emb=True)
        experts.append(model)
        models_list.append(model_type)

    # LOAD GATING NETWORK
    if 'cat' in checkpoint_moe.lower():
        moe_model = MOE_ICASSP(experts=experts, return_weights=False)
    else:
        moe_model = MOE_Way2(experts=experts, return_weights=False)

    for expert in experts:
        expert.to(device)

    moe_model = (moe_model).to(device)

    moe_model.load_state_dict(torch.load(checkpoint_moe, map_location=device))
    print('Model loaded : {}'.format(os.path.basename(checkpoint_moe)))

    # RUN EVALUATION
    if config['eval_all']:
        result_path = os.path.join(save_folder, "ALL", checkpoint_name)
        run_eval(moe_model, models_list, result_path, config)
    else:
        targets = config['selected']
        for target in targets:
            config['selected'] = [target]
            result_path = os.path.join(save_folder, target, checkpoint_name)
            run_eval(moe_model, models_list, result_path, config)


if __name__ == '__main__':

    config = read_yaml(Path(__file__).parent.parent / "config" / "args_params.yaml")
    config_paths = read_yaml(Path(__file__).parent.parent / "config" / "args_paths.yaml")
    config.update(config_paths)
    del config_paths

    seed_everything(config['seed'])

    ckpt_exp_path = os.path.join(config['save_model_folder'], 'experts')
    ckpt_moe_path = os.path.join(config['save_model_folder'], 'MoE')
    save_path = os.path.join(config['save_results_folder'], 'MoE')

    # --------------------- PLEASE DEFINE ----------------------#

    config['eval_all'] = False

    config['sanity_check'] = False
    config['target_check'] = True

    config['train_data'] = 'asv5'
    config['test_data'] = 'asv5'

    # ckpt_moe = os.path.join(ckpt_moe_path, 'frozen_5_exp_new_calib_fixed.pth')   # best model trained on asv19
    # ckpt_moe = os.path.join(ckpt_moe_path, 'asv5_frozen_5_exp_calib.pth')      # counterpart trained on asv5
    ckpt_moe = os.path.join(ckpt_moe_path, 'asv5_frozen_5_exp_linear_comb.pth')

    # ------------------------- Thanks -------------------------#

    if config['train_data'] == 'asv19':
        ckpts_experts = [
            os.path.join(ckpt_exp_path, 'FLAT_PITCH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'F0_INCONS_MLP.pth'),
            os.path.join(ckpt_exp_path, 'OVERSMOOTH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'MIN_PHASE_LCNN.pth'),
            os.path.join(ckpt_exp_path, 'ENERGY_MLP.pth')
        ]
    else:
        ckpts_experts = [
            os.path.join(ckpt_exp_path, 'ASV5_FLAT_PITCH_MLP.pth'),
            os.path.join(ckpt_exp_path, 'ASV5_F0_INCONS_MLP.pth'),
            os.path.join(ckpt_exp_path, 'ASV5_OVERSMOOTH_GRU.pth'),
            os.path.join(ckpt_exp_path, 'ASV5_MIN_PHASE_LCNN.pth'),
            os.path.join(ckpt_exp_path, 'ASV5_ENERGY_LCNN.pth')
        ]

    if config['target_check']:
        if config['test_data'] == 'asv19':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
        elif config['test_data'] == 'asv5':
            config['selected'] = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A14']
        elif config['test_data'] == 'wf':
            config['selected'] = ["tts_ljspeech", "ljspeech_full_band_melgan", "ljspeech_hifiGAN", "ljspeech_melgan",
                                  "ljspeech_melgan_large", "ljspeech_multi_band_melgan", "ljspeech_parallel_wavegan",
                                  "ljspeech_waveglow"]  #, "jsut_multi_band_melgan", "jsut_parallel_wavegan"]
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
        save_folder = os.path.join(save_path, config['test_data'], "TargetCheck")

    else:
        config['selected'] = ['flat_pitch', 'f0_incons', 'oversmooth', 'min_phase', 'energy']
        save_folder = os.path.join(save_path, config['test_data'], "SanityCheck")

    print("\nEVALUATION MODE\n")
    print(f"Is sanity check? {config['sanity_check']}")
    print(f"Is target check? {config['target_check']} \n")

    test_moe(ckpt_moe, ckpts_experts, save_folder, config)




