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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device : {device}")

TIME = strftime("%y%m%d_%H%M%S", localtime())
DBDIR = "/home/aicontest/DF/data/audio"
DBNAME = "all_more_train"
TNAME = "test"
SUBMIT = "/home/aicontest/DF/result/uijin"
SAVE = "/home/aicontest/DF/parameter/uijin"
#TESTING = 100

########################################################################
######################### argument 설정 #################################
########################################################################

def get_argument():
    parser = argparse.ArgumentParser(description="aicontest")

    # directory
    parser.add_argument("-t", "--time", type=str, default=TIME, help="invoke time to utilize as a prefix")
    parser.add_argument("-db", "--db_dir", default=DBDIR, type=str, help="dataset directory")
    parser.add_argument("-dname", "--db_name", default=DBNAME, type=str, help="train dataset name (csv file)")
    parser.add_argument("-tname", "--test_name", default=TNAME, type=str, help="test dataset name (csv file)")
    parser.add_argument("-sb", "--submit", default=SUBMIT, type=str, help="submit file root")
    parser.add_argument("-sm", "--save_model", default=SAVE, type=str, help="save model parameter root")
    
    # inference
    parser.add_argument("-val", "--validation", action="store_true", help="split images as validation set")
    parser.add_argument("-es", "--early_stop", action="store_true", help="enable early stopping when validation enabled")
    parser.add_argument("-esp", "--early_stop_patient", type=int, default=5 ,help="enable early stopping when validation enabled")
    parser.add_argument("-vs", "--validation_size", type=int, default=0.2, help="validation percent")
    
    #hyperparameter
    parser.add_argument("--config", dest="config", type=str, help="configuration file", required=True)
    parser.add_argument("-sd", "--seed", type=int, default=42, help="fixed seed number")
    parser.add_argument("-nep", "--n_epoch", type=int, default=20, help="number of epochs (maximum when early stopping enabled)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("-pf", "--posfix", type=str, default="", help="postfix for checkpt")
    parser.add_argument("-sf", "--sample_frequency", type=int, default=16000, help="Resample the waveform using sample rate")
    parser.add_argument("-svt", "--sound_vector_threshold", type=int, default=80000, help="fixed sound vector size")


    parser.add_argument('--algo', type=int, default=4, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    return parser.parse_args()

args = get_argument()
print("== argment 처리 완료 ==")

with open(args.config, "r") as f_json:
    config = json.loads(f_json.read())
model_config = config["model_config"]
# model_config["first_conv"] = args.sound_vector_threshold

########################################################################
##################### 편한 실험을 위한 seed 고정 ############################
########################################################################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(args.seed)
print("== seed 고정 완료 ==")

########################################################################
########################### 데이터 받기 ###################################
########################################################################

df = pd.read_csv(args.db_dir + '/' + args.db_name + '.csv')
df = df.sample(frac=1).reset_index(drop=True).iloc[:50000]
print(df.head())
print(f"데이터 길이 : {len(df)}")
print("== 데이터 받기 완료 ==")

## validation 처리
if args.validation:
    train, val = train_test_split(df, test_size=args.validation_size, random_state=args.seed, stratify=df[['real', 'fake']])
else:
    train = df
print("== validation 처리 완료 ==")

########################################################################
#################### Front-end : audio feature 뽑기 #####################
########################################################################

#front_model = wav2vec2_base().to(device)

## vector 길이 맞추기
def pad_or_truncate(features, device, fixed_length=args.sound_vector_threshold, min_second_dim=3):
    #print(f"니냐? {features.shape[0]}")
    if features.shape[0] >= fixed_length:
        # Truncate the tensor to the fixed length
        features = features[:fixed_length]
    else:
        # Repeat the tensor to the fixed length
        num_repeats = (fixed_length // features.shape[0]) + 1
        padded_features = torch.tensor([])

        for _ in range(num_repeats):
            padded_features = torch.cat((padded_features, features), dim=0)

        #print(f"누가 범인인가 {padded_features.shape[0]}")
        features = padded_features[:fixed_length]

    
    return features
    
def front_end(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 이부분에 frontend 삽입
        waveform, sample_rate = torchaudio.load(args.db_dir + '/' + row['path'])
        
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.sample_frequency)
        waveform = resampler(waveform)

        waveform = pad_or_truncate(waveform.squeeze(0), device)
        #print(waveform.shape[0]) 
        features.append(waveform)

        if train_mode:
            label_vector = np.zeros(2, dtype=float)
            label_vector[1] = row['real']
            label_vector[0] = row['fake']
            labels.append(label_vector)
        # 임시 과정 보기 위함
        # if len(features) >= 100:
        #      break

    if train_mode:
        return features, labels
    return features

train_data, train_labels = front_end(train, True)
print("== train front end 완료 ==")
if args.validation:
    val_data, val_labels = front_end(val, True)
    print("== validation front end 완료 ==")

########################################################################
########################### Dataset ####################################
########################################################################

def get_module(directory, module_file , module_name):
    sys.path.append(directory)
    module = import_module(module_file)
    function = getattr(module, module_name)

    return function

class CustomDataset(Dataset):
    def __init__(self, data, label, args, algo=4, noise_rate=0.1, noise_probability=0.1):
        self.data = data
        self.label = label
        self.args = args
        self.algo = algo 
        self.noise_rate = noise_rate
        self.noise_probability = noise_probability
        self.process_Rawboost_feature = get_module('/home/aicontest/DF/modules','augmentation', 'process_Rawboost_feature')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        # 이쪽에 전처리, augmentation

        if self.label is not None:

            # # 특정 비율의 데이터에만 노이즈 추가
            # if np.random.rand() < self.noise_probability:
            #     wn = np.random.randn(len(data_item))
            #     data_item = data_item + self.noise_rate * wn

            data_item = self.process_Rawboost_feature(self.data[index], self.args.sample_frequency, self.args, self.algo)

            return data_item, self.label[index]
        return self.data[index]

train_dataset = CustomDataset(train_data, train_labels, args, args.algo)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

if args.validation:
    val_dataset = CustomDataset(val_data, val_labels, args, args.algo)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )


print("== pre-processing 완료 ==")

########################################################################
################## back-end : audio classification #####################
########################################################################


def train(model, optimizer, train_loader, device, val_loader=False):
    model.to(device)
    #weight = torch.FloatTensor([0.1, 0.9]).to(device)
    #criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    patient = 0

    for epoch in range(1, args.n_epoch+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader), total=len(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(features)
            #output = output[1]
            output = torch.sigmoid(output[1])
            #print(f'Train output: {output}')
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)

        if val_loader != False:          
            _val_loss, _val_score = validation(model, criterion, val_loader, device)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}], patient: [{patient}]')
            
            if best_val_score < _val_score:
                best_val_score = _val_score
                best_model = model
                if args.early_stop:
                    patient = 0
            elif args.early_stop:
                patient +=1
                if patient >= args.early_stop_patient:
                    break
        else:
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}]')
            best_model = model
    
    if best_model is None:
        best_model = model

    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            probs = torch.sigmoid(probs[1])
            #print(f'Validation probs: {probs}')
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score

def get_model(model_config: Dict, device: torch.device):
    model_directory = "/home/aicontest/DF/models"
    sys.path.append(model_directory)
    """Define DNN model architecture"""
    module = import_module(model_config["architecture"])
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

# # 추가 디버깅용 출력 함수
# def debug_model_output(model, data_loader, device):
#     with torch.no_grad():
#         for features, labels in data_loader:
#             features = features.float().to(device)
#             output = model(features)
#             output = F.softmax(output[1], dim=1)
#             print(f'Debug output: {output}')
#             break  # 첫 번째 배치만 확인

model = get_model(model_config, device)
#model.load_state_dict(torch.load(config["model_path"], map_location=device))
optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)

if args.validation:
    infer_model = train(model, optimizer, train_loader, device, val_loader)
else:
    infer_model = train(model, optimizer, train_loader, device)

# # 학습 모드에서 출력 확인
# print("Training mode output check:")
# infer_model.train()
# debug_model_output(infer_model, train_loader, device)

# # 평가 모드에서 출력 확인
# print("Evaluation mode output check:")
# infer_model.eval()
# debug_model_output(infer_model, val_loader, device)

torch.save(infer_model.state_dict(), args.save_model + "/result_" + str(args.time) + ".pt")
print("== back_end 완료 ==")

########################################################################
########################### inference ##################################
########################################################################

test = pd.read_csv(args.db_dir + '/' + args.test_name + '.csv')
test_data = front_end(test, False)
test_dataset = CustomDataset(test_data, None, args)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

def inference(model, test_loader, device):
    model.to(device)
    model.train()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader), total=len(test_loader)):
            features = features.float().to(device)
            
            probs = model(features)
            #probs = probs[1]
            probs = torch.sigmoid(probs[1])

            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

preds = inference(infer_model, test_loader, device)
print("== inference 완료 ==")

########################################################################
########################## submission ##################################
########################################################################

submit = pd.read_csv(args.db_dir + '/' + 'sample_submission.csv')
submit.iloc[:, 1:] = preds
#submit.iloc[:TESTING, 1:] = preds
print(submit.head())

########################################################################
######################## noise detection ###############################
########################################################################

non_speech_list = []

with open("../non_speech.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        file_path = line.split(' ')[0]
        file_name = file_path.split('/')[-1][:-4]
        non_speech_list.append(file_name)
        

submit.loc[submit['id'].isin(non_speech_list), ['fake', 'real']] = 0
submit.to_csv(args.submit + "/submit_" + str(args.time) + ".csv" , index=False)
print("== submit 완료 ==")