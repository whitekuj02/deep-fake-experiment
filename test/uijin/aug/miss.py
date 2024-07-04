from pydub import AudioSegment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math


df = pd.read_csv("/home/aicontest/DF/data/audio/mix_train.csv")

df['path'] = ["."+x[29:]for x in df['path']]

new_df = pd.DataFrame({
    "id": df['id'],
    "path": df['path'] ,
    "real": df['real'],
    "fake": df['fake']
})

new_df.to_csv("/home/aicontest/DF/data/audio/mix_train.csv",index=False)