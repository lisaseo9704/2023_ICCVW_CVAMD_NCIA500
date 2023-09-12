import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import cv2
import pickle
import pandas as pd
import random

from utils.util import *

class dataset_mimic(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None, metafile_path=None):
        super(dataset_mimic, self).__init__()
        self.root = root
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(metafile_path)[:100]
        self.img_path = df['path'].values
        self.target = df.drop('path', axis=1).values

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.root + '/' + self.img_path[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR) # do not use cv2.IMREAD_GRAYSCALE
        # img = np.expand_dims(img, axis=2) # size : HxWx1
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.) # size : 1xHxW

        target = self.target[index]

        return (img, self.inp), target
    
class dataset_mimic_aug(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None, metafile_path=None):
        super(dataset_mimic_aug, self).__init__()
        self.root = root
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(metafile_path)
        self.img_path = df['path'].values
        self.target = df.drop('path', axis=1).values

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.root + '/' + self.img_path[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR) # do not use cv2.IMREAD_GRAYSCALE
        # img = np.expand_dims(img, axis=2) # size : HxWx1
        mode = random.randrange(12)
        img = augment_img(img, mode)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).clamp_(0, 1) # size : 1xHxW

        target = self.target[index]

        return (img, self.inp), target
    
class dataset_submit(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None, metafile_path=None):
        super(dataset_submit, self).__init__()
        self.root = root
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(metafile_path)
        self.img_path = df['path'].values

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.root + '/' + self.img_path[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR) # do not use cv2.IMREAD_GRAYSCALE
        # img = np.expand_dims(img, axis=2) # size : HxWx1
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.) # size : 1xHxW

        return (img, self.inp), self.img_path[index]