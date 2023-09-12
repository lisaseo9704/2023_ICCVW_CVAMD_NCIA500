#%%
import pandas as pd
import numpy as np
import cv2

df1 = pd.read_csv('./data/mimic/train_split.csv')
df2 = pd.read_csv('./data/mimic/val_split.csv')
df1 = df1['path'].values
df2 = df2['path'].values
df = np.concatenate((df1, df2))
print(df, df.shape, sep='\n')
# %%
import os
from tqdm import tqdm

img_root = '/data1/data_500/mimic_jpg'
save_root = '/data1/data_500/resized_1024/files'
for i in tqdm(range(len(df))):
    path = df[i]
    path = path.split('/')[1:]
    img_path = img_root
    save_path = save_root
    for item in path:
        img_path = img_path + '/' + item
        if item != path[-1]:
            save_path = save_path + '/' + item
        else:
            basename = item
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024))
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, basename), img)
# %%
import pandas as pd
import numpy as np
import cv2

df = pd.read_csv('../data/mimic/development.csv')
# %%
val = df.values
#%%
print(val[:,5])
# %%
import os
from tqdm import tqdm

img_root = '/data1/ICCV_data/mimic_jpg'
save_root = '/data1/ICCV_data/resized_1024/files'
for i in tqdm(range(len(val))):
    path = val[i, 5]
    path = path.split('/')[1:]
    img_path = img_root
    save_path = save_root
    for item in path:
        img_path = img_path + '/' + item
        if item != path[-1]:
            save_path = save_path + '/' + item
        else:
            basename = item
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img_path)
    img = cv2.resize(img, (1024, 1024))
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, basename), img)
# %%
df.head()
# %%
