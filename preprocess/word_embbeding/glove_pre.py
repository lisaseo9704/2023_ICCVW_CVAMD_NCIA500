#%%
import pandas as pd

item_list = ['Atelectasis', 'Calcification_of_the_Aorta', 'Cardiomegaly', 
             'Consolidation', 'Edema', 'Emphysema',
             'Enlarged_Cardiomediastinum', 'Fibrosis', 'Fracture', 
             'Hernia', 'Infiltration', 'Lung_Lesion', 'Lung_Opacity', 'Mass', 
             'No_Finding', 'Nodule', 'Pleural_Effusion', 'Pleural_Other',
             'Pleural_Thickening', 'Pneumoperitoneum', 'Pneumonia', 
             'Pneumomediastinum','Pneumothorax', 'Subcutaneous_Emphysema',
             'Support_Devices', 'Tortuous_Aorta']

df = pd.read_csv('/data1/home/min356/workspace/ML-GCN/data/mimic/train_train.csv')
df = df.drop(['path'], axis=1)
# %%
for i in range(len(item_list)):
    print(item_list[i])
# %%
sentences = []

for i in range(len(df)):
    temp_sentences = []
    for j in range(len(df.loc[i])):
        if df.loc[i][j] == 1:
            temp_sentences.append(item_list[j])
    sentences.append(temp_sentences)
# %%
print(sentences)
# %%
print(len(sentences))
# %%
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index)
print('단어 집합 :',vocab_size)
# %%
X_encoded = tokenizer.texts_to_sequences(sentences)
print('정수 인코딩 결과 :',X_encoded)
#%%
max_len = max(len(l) for l in X_encoded)
print('최대 길이 :', max_len)
#%%
X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
print('패딩 결과 :')
print(X_train)
#%%
print(type(X_train))
#%%
co_occur = df.T.dot(df)
#%%
from mittens import GloVe

glove_model = GloVe(n=300, max_iter=10000)
embeddings = glove_model.fit(co_occur)

# %%
print(embeddings, embeddings.shape)

# %%
import pickle

with open('/data1/home/min356/workspace/ML-GCN/data/mimic/mimic_glove_word2vec.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# %%
with open('/data1/home/min356/workspace/ML-GCN/data/mimic/mimic_glove_word2vec.pkl', 'rb') as f:
    y = pickle.load(f)
# %%
print(y)
# %%
