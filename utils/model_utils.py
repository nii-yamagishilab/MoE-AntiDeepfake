import os
from models.small_models import *
from models.lcnn_model import LCNN

                            # ----------------------------------------#
                            # WARNING: THE ORDERING LOGIC MATTERS !!! #
                            # ----------------------------------------#

MODEL_REGISTRY = [
    ("MLP", MLP_Model),      # flat pitch
    ("MLP", MLP_Model_v2),   # f0 inconsistency
    ("GRU", GRU_Model),
    ("LCNN", LCNN),
    ("LCNN", LCNN)
]

MANIPULATIONS_REGISTRY = [
    ("MLP", "flat_pitch"),
    ("MLP", "f0_incons"),
    ("GRU", "oversmooth"),
    ("LCNN", "min_phase"),
    ("LCNN", "energy")
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
