import librosa
import sys
import os

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import json

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchaudio.models import wav2vec2_base
from torchaudio.models import hubert_base
from sklearn.metrics import roc_auc_score
from torch import Tensor

import torch
import torchaudio
import torchmetrics
import os
import argparse
from time import time, strftime, localtime
#from modules.get_model import get_model
from typing import Dict
from importlib import import_module

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


waveform, sample_rate = torchaudio.load("/home/aicontest/DF/data/audio/train/AAACWKPZ.ogg")
print(waveform)
print(waveform.shape)
print("================================")

resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)
print(waveform)
print(waveform.shape)
print("================================")

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

waveform = waveform / waveform.abs().max()
print(waveform)
print(waveform.shape)
print("================================")

front_model = wav2vec2_base().to(device)


def pad_or_truncate(features, device, fixed_length=128):
    if features.shape[0] > fixed_length:
        features = features[:fixed_length, :]
    else:
        padding = torch.zeros((fixed_length - features.shape[0], features.shape[1])).to(device)
        features = torch.cat((features, padding), dim=0)
    
    return features

with torch.no_grad(): # 이건 pre-trained 되어 있어서 학습 안하나 봄, 앙상블은 간단하게 여기 Model 하나 더 추가
    output = front_model(waveform.to(device))
    print(output[0].squeeze(0).shape)
    _output = pad_or_truncate(output[0].squeeze(0), device) # 이거 flatten 인데 임시적인 방안
    print(_output.shape)
    print("================================")

# ================================================================================
