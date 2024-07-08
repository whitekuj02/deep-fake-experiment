from pydub import AudioSegment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

# sound1 = AudioSegment.from_ogg("/home/aicontest/DF/data/audio/train/AAACWKPZ.ogg")
# sound2 = AudioSegment.from_ogg("/home/aicontest/DF/data/audio/train/AAAQOZYI.ogg")

# # mix sound2 with sound1, starting at 5000ms into sound1)
# output = sound1.overlay(sound2, position=0)

# # save the result
# output.export("./mixed_sounds.wav", format="wav")

df = pd.read_csv("/home/aicontest/DF/data/audio/train.csv")

db_dir = "/home/aicontest/DF/data/audio"
print(df.head())



# data['real'] = data['label'].apply(lambda x: 1 if x == 'real' else 0)
# data['fake'] = data['label'].apply(lambda x: 1 if x == 'fake' else 0)

train_1, train_2, _, _ = train_test_split(df, df['label'], test_size=0.5, random_state=42, shuffle=True)

id = []
path = []
real = []
fake = []


for (_, row_1), (_, row_2) in tqdm(zip(train_1.iterrows(), train_2.iterrows()), total=len(train_1)):
    
    sound1 = AudioSegment.from_ogg(db_dir + '/' + row_1["path"])
    sound2 = AudioSegment.from_ogg(db_dir + '/' + row_2["path"])
    
    output = sound1.overlay(sound2, position=0)
    new_path = db_dir + '/mix_train/' + row_1["id"] + '_' + row_2['id'] + '.wav'
    output.export(new_path, format="wav")

    id.append(row_1["id"] + '_' + row_2['id'])
    path.append("./mix_train/" + row_1["id"] + '_' + row_2['id'] + '.wav')

    label_1 = row_1['label']
    label_2 = row_2['label']
    label_vector_1 = np.zeros(2, dtype=float)
    label_vector_2 = np.zeros(2, dtype=float)
    label_vector_1[0 if label_1 == 'fake' else 1] = 1
    label_vector_2[0 if label_2 == 'fake' else 1] = 1
    
    label_vector = [ int(min(x+y,1)) for (x,y) in zip(label_vector_1, label_vector_2)]
    real.append(label_vector[0])
    fake.append(label_vector[1])

   
aug_data = pd.DataFrame({
    "id": id,
    "path": path,
    "real": real,
    "fake": fake
})

print(aug_data.head())

aug_data.to_csv(db_dir + "/mix_train.csv", index=False)
