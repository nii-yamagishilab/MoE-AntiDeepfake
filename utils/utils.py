import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score

from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_metrics_v2(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr

    # Interpolate
    eer_func = interp1d(fpr, fnr)
    eer = brentq(lambda x: eer_func(x) - x, 0., 1.)

    return rocauc, eer

def compute_metrics(labels, scores):
    fpr, tpr, thres = roc_curve(labels, scores)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    # optimal_index = np.argmax(tpr - fpr)
    # optimal_threshold = thres[optimal_index]
    # binary_pred = np.where(scores < -0.5, 0, 1)
    # acc = balanced_accuracy_score(labels, binary_pred)

    return rocauc, eer

def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
