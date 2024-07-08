from pydub import AudioSegment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math


df = pd.read_csv("/home/aicontest/DF/data/audio/train.csv")
df_2 = pd.read_csv("/home/aicontest/DF/data/audio/more_mix_train.csv")

print(df.head())
id = pd.concat([df["id"], df_2["id"]], ignore_index=True)
path = pd.concat([df["path"], df_2["path"]], ignore_index=True)
real = pd.concat([df['label'].apply(lambda x: 1 if x == 'real' else 0), df_2["real"]], ignore_index=True)
fake = pd.concat([df['label'].apply(lambda x: 1 if x == 'fake' else 0), df_2["fake"]], ignore_index=True)

new_df = pd.DataFrame({
    "id": id,
    "path": path,
    "real": real,
    "fake": fake
})

new_df.to_csv("/home/aicontest/DF/data/audio/all_more_train.csv",index=False)