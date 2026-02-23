import os
from models.small_models import *
from models.lcnn_model import LCNN

MODEL_REGISTRY = [
    ("MLP", MLP_Model),
    ("MLP", MLP_Model_v2),
    ("GRU", GRU_Model),
    ("LCNN", LCNN),
    ("LCNN", LCNN)
]

MANIPULATIONS_REGISTRY = [
    ("MLP", "f0_incons"),
    ("MLP", "flat_pitch"),
    ("GRU", "oversmooth"),
    ("LCNN", "min_phase"),
    ("LCNN", "perf_sil")
]

TARGET_DF_REGISTRY = [
    # ("f0_incons", "A04"),
    # ("oversmooth", "A03"),
    # ("min_phase", "A03"),
    ("flat_pitch", "CUSTOM")
    # perf_sil,  NO
]

def get_model_type_from_ckpt(ckpt_path):
    filename = os.path.basename(ckpt_path).upper()
    name_no_ext = os.path.splitext(filename)[0]

    tokens = name_no_ext.split("_")
    model_names = [name for name, _ in MODEL_REGISTRY]

    for token in tokens:
        token_upper = token.upper()
        if token_upper in model_names:
            return token_upper

    raise ValueError(f"No model recognized from checkpoint: {filename}")