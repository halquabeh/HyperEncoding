import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
from models import VGG


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def BPTT_attack(model, image, T):
    output = model(image).mean(0)
    return output

def BPTR_attack(model, image, T):
    model.set_simulation_time(T, mode='bptr')
    output = model(image).mean(0)
    model.set_simulation_time(T)
    return output

def Act_attack(model, image, T):
    model.set_simulation_time(0)
    output = model(image)
    model.set_simulation_time(T)
    return output

def create_model(model_name, encoding, time, num_labels, znorm,encode_in):
    if 'vgg' in model_name:
        model = VGG(model_name, encoding, time, num_labels, znorm,3,encode_in)
    else:
        raise AssertionError("model not supported")
    return model