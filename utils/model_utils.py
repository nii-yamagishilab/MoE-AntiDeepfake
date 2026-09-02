import os
from models.small_models import *
from models.lcnn_model import LCNN

                            # ----------------------------------------#
                            # WARNING: THE ORDERING LOGIC MATTERS !!! #
                            # ----------------------------------------#

MODEL_REGISTRY = [
    ("MLP", MLP_F0_Incons),
    ("MLP", MLP_Smooth),
    ("LCNN", LCNN),
    ("MLP", MLP_Energy),
    ("MLP", MLP_Checkerboard),
    # ("MLP", MLP_Checkerboard)
]

MANIPULATIONS_REGISTRY = [
    ("MLP", "f0_incons"),
    ("MLP", "oversmooth"),
    ("LCNN", "min_phase"),
    ("MLP", "energy"),
    ("MLP", "tonal"),
    # ("MLP", "filtering")
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
