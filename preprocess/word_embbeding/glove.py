#%%
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['SupportDevices', 'LungOpacity', 'Cardiomegaly', 'PleuralEffusion',
             'Atelectasis', 'Pneumonia', 'NoFinding', 'Edema', 'EnlargedCardiomediastinum',
             'Consolidation', 'Pneumothorax', 'Fracture' , 'Infiltration', 'Nodule',
             'Mass', 'CalcificationoftheAorta', 'Emphysema', 'Hernia', 'TortuousAorta',
             'PleuralThickening', 'SubcutaneousEmphysema', 'LungLesion',
             'Fibrosis', 'Pneumomediastinum', 'PleuralOther', 'Pneumoperitoneum']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index)

#%%
print('단어 집합 :',vocab_size)
X_encoded = tokenizer.texts_to_sequences(sentences)
print('정수 인코딩 결과 :',X_encoded)
X_train = pad_sequences(X_encoded, maxlen=1, padding='post')
print('패딩 결과 :')
print(X_train)
#%%
embedding_dict = dict()

f = open('glove.6B.50d.txt', encoding="utf8")

for line in f:
    word_vector = line.split()
    word = word_vector[0]

    # 100개의 값을 가지는 array로 변환
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
    embedding_dict[word] = word_vector_arr
f.close()

print('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))

#%%
print(embedding_dict.get('subcutaneous'))
# %%
print(1)
# %%
