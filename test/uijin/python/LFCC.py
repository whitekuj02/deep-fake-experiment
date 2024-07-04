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
from torchaudio.transforms import LFCC
import torchvision.models as models

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
DBNAME = "all_train"
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
    #parser.add_argument("--config", dest="config", type=str, help="configuration file", required=True)
    parser.add_argument("-sd", "--seed", type=int, default=42, help="fixed seed number")
    parser.add_argument("-nep", "--n_epoch", type=int, default=50, help="number of epochs (maximum when early stopping enabled)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("-pf", "--posfix", type=str, default="", help="postfix for checkpt")
    parser.add_argument("-sf", "--sample_frequency", type=int, default=100000, help="Resample the waveform using sample rate")
    parser.add_argument("-svt", "--sound_vector_threshold", type=int, default=512, help="fixed sound vector size")
    return parser.parse_args()

args = get_argument()
print("== argment 처리 완료 ==")

# with open(args.config, "r") as f_json:
#     config = json.loads(f_json.read())
# model_config = config["model_config"]
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

front_model = wav2vec2_base().to(device)

## vector 길이 맞추기
def pad_or_truncate(features, device, fixed_length=args.sound_vector_threshold, min_second_dim=3):
    if features.shape[1] > fixed_length:
        features = features[:,:fixed_length]
    else:
        padding = np.zeros((features.shape[0], fixed_length - features.shape[1]))
        features = np.concatenate((features, padding), axis=1)
    
    return features
    
def front_end(df, train_mode=True):
    features = []
    labels = []
    transform = LFCC(
        sample_rate=args.sample_frequency,
        n_lfcc=13,
        speckwargs={"n_fft": 400, "hop_length": 160, "center": False}).to(device)

    for _, row in tqdm(df.iterrows()):
        # librosa패키지를 사용하여 wav 파일 load
        y, sr = torchaudio.load(args.db_dir + "/" +row['path'], normalize=True)
        
        # librosa패키지를 사용하여 mfcc 추출
        lfcc = transform(y.to(device)).squeeze(0).cpu().numpy()
        lfcc = pad_or_truncate(lfcc, device)
        #lfcc = np.mean(lfcc.T, axis=0)
        features.append(lfcc)

        if train_mode:
            #label = row['label']
            label_vector = np.zeros(2, dtype=float)
            #label_vector[0 if label == 'fake' else 1] = 1
            label_vector[1] = row['real']
            label_vector[0] = row['fake']
            labels.append(label_vector)

        # if len(features) >= 10:
        #     break
        
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

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        # 이쪽에 전처리, augmentation
        
        if self.label is not None:
            return self.data[index], self.label[index]
        return self.data[index]

train_dataset = CustomDataset(train_data, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

if args.validation:
    val_dataset = CustomDataset(val_data, val_labels)
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
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    patient = 0

    for epoch in range(1, args.n_epoch+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(features)
            #output = F.softmax(output, dim=1)
            output = torch.sigmoid(output)
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
            #probs = F.softmax(probs, dim=1)
            probs = torch.sigmoid(probs)
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

def get_model(device: torch.device):
    model_directory = "/home/aicontest/DF/models"
    sys.path.append(model_directory)
    """Define DNN model architecture"""
    module = import_module("Res2Net")
    _model = getattr(module, "res2net50")
    model = _model(pretrained=True).to(device)
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

#model = get_model(model_config, device)
#model.load_state_dict(torch.load(config["model_path"], map_location=device))
#model = get_model(device)
class CustomResNetModel(nn.Module):
    def __init__(self, input_channels=1, output_dim=2):
        super(CustomResNetModel, self).__init__()
        # Pretrained ResNet18 모델 로드
        self.resnet = models.resnet18(pretrained=True)
        
        # 첫 번째 레이어 수정: ResNet은 기본적으로 3채널 이미지를 입력으로 받지만, 여기서는 input_channels 사용
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Fully Connected Layer 수정
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_dim)
    
    def forward(self, x):
        # x의 shape은 (batch_size, 13, 512)
        # ResNet은 4D 텐서를 기대하므로 (batch_size, 13, 1, 512)로 변형
        x = x.unsqueeze(1)  # (batch_size, 13, 1, 512)
        
        x = self.resnet(x)
        return x
    
model = CustomResNetModel().to(device)
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

#torch.save(infer_model.state_dict(), args.save_model + "/result_" + str(args.time) + ".pt")
print("== back_end 완료 ==")

########################################################################
########################### inference ##################################
########################################################################

test = pd.read_csv(args.db_dir + '/' + args.test_name + '.csv')
test_data = front_end(test, False)
test_dataset = CustomDataset(test_data, None)
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
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            
            probs = model(features)
            #probs = F.softmax(probs, dim=1)
            probs = torch.sigmoid(probs)

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
submit.head()

submit.to_csv(args.submit + "/submit_" + str(args.time) + ".csv" , index=False)
print("== submit 완료 ==")