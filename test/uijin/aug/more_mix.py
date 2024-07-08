from pydub import AudioSegment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

df = pd.read_csv("/home/aicontest/DF/data/audio/train.csv")

db_dir = "/home/aicontest/DF/data/audio"

train_1 = df.copy()
train_2 = df.copy().sample(frac=1).reset_index(drop=True)  # 열 순서 바꾸기

print(train_1.head())
print(train_2.head())

print(f"train_1 length : {len(train_1)}, train_2 length : {len(train_2)}")

id = []
path = []
real = []
fake = []

sample_num = 0
skip = 0
sample_threshold = 4
already_mix = []
already_mix_array = []

# 초기 빈 CSV 파일 생성 (헤더 포함)
aug_data_path = db_dir + "/more_mix_train.csv"
pd.DataFrame(columns=["id", "path", "real", "fake"]).to_csv(aug_data_path, index=False)

with tqdm(total=len(train_1), desc='Processing', ncols=100) as pbar:
    for (_, row_1) in train_1.iterrows():
        for (_, row_2) in train_2.iterrows():
            if row_1['path'] == row_2['path']:  # 같은 path거나 이미 있는 mix 정보이면
                skip += 1
                continue

            if row_1["id"] + '_' + row_2['id'] in already_mix or row_2["id"] + '_' + row_1['id'] in already_mix:
                skip += 1
                continue

            sound1 = AudioSegment.from_ogg(db_dir + '/' + row_1["path"])
            sound2 = AudioSegment.from_ogg(db_dir + '/' + row_2["path"])

            output = sound1.overlay(sound2, position=0)
            new_path = db_dir + '/more_mix_train/' + row_1["id"] + '_' + row_2['id'] + '.wav'
            output.export(new_path, format="wav")

            new_id = row_1["id"] + '_' + row_2['id']
            new_path_relative = "./more_mix_train/" + row_1["id"] + '_' + row_2['id'] + '.wav'
            id.append(new_id)
            path.append(new_path_relative)

            label_1 = row_1['label']
            label_2 = row_2['label']
            label_vector_1 = np.zeros(2, dtype=float)
            label_vector_2 = np.zeros(2, dtype=float)
            label_vector_1[0 if label_1 == 'fake' else 1] = 1
            label_vector_2[0 if label_2 == 'fake' else 1] = 1

            label_vector = [int(min(x + y, 1)) for (x, y) in zip(label_vector_1, label_vector_2)]
            real.append(label_vector[0])
            fake.append(label_vector[1])

            already_mix.append(new_id)
            already_mix_array.append([row_1["id"], row_2["id"]])
            sample_num += 1

            # 데이터 프레임 생성 및 실시간 저장
            aug_data = pd.DataFrame({
                "id": [new_id],
                "path": [new_path_relative],
                "real": [label_vector[0]],
                "fake": [label_vector[1]]
            })
            aug_data.to_csv(aug_data_path, mode='a', header=False, index=False)

            if sample_num >= sample_threshold:
                sample_num = 0
                break

        train_2 = train_2.sample(frac=1).reset_index(drop=True)  # 재 배열
        pbar.update(1)
        pbar.set_postfix(skip=skip, mix_len=len(already_mix_array))

print(f"skip : {skip}")
print(f"mix len : {len(already_mix_array)}, duplicate : {len(already_mix_array) - len(list(set(tuple(map(tuple, already_mix_array)))))}")

print(pd.read_csv(aug_data_path).head())
