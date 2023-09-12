import argparse
import cv2
import glob
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

from models.ml_gcn_me import *
from data.dataset_mimic import dataset_submit



def main(args):
    # set up GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5,6"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # make output folder
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # set up model
    checkpoint = torch.load(args.checkpoint)
    model = gcn_resnet101(num_classes=args.num_class, t=0.4, adj_file=args.adj, block=Bottleneck)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = nn.DataParallel(model, device_ids=args.gpu_ids).to(device=device)

    # set dataset
    dataset = dataset_submit(args.data, inp_name = args.inp, metafile_path=args.metafile)
    dataloader = DataLoader(dataset, batch_size = args.batch, num_workers=16, shuffle=False)
    total_num = dataset.__len__()

    # make save dataframe
    item_list = ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 
                'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 
                'Fibrosis', 'Fracture','Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 
                'Mass', 'No Finding', 'Nodule', 'Pleural Effusion', 'Pleural Other', 
                'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum',
                'Pneumothorax', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta']
    item_list.insert(0,'path')
    df = pd.read_csv(args.metafile)
    res = df.drop(['path'], axis=1)
    save = df['path'].values
    buffer = torch.from_numpy(np.zeros((total_num, args.num_class)))

    # start test
    print(f'Start test. Total number of test data is {total_num}.')
    for _, ((feature, inp), path) in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            feature = feature.to(device)
            inp = inp.to(device)
            output = (model(feature, inp)['output']).sigmoid()
            for i in range(len(path)):
                save_idx = np.where(save == path[i])[0][0]
                buffer[save_idx] = output[i]
    buffer.numpy()
    
    # save
    save = np.expand_dims(save, axis=1)
    save = np.concatenate([save, buffer], axis=1)
    save = pd.DataFrame(save, columns=item_list)
    result = pd.concat([res, save], axis=1)
    result.to_csv(args.save_path, index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model infos
    parser.add_argument('--checkpoint', type=str, default='checkpoint/mimic_tr/model_best_0.2883.pth.tar', help='model path.')
    parser.add_argument('--num_class', type=int, default=26, help='number of class for classification')
    parser.add_argument('--inp', type=str, default='data/mimic/mimic_glove_word2vec.pkl', help='path for word embedding')
    parser.add_argument('--adj', type=str, default='data/mimic/mimic_adj.pkl', help='path of adjacency matrix')

    # meta infos
    parser.add_argument('--data', default='/data1/ICCV_data/resized_1024_test', metavar='DIR', help='path to dataset (e.g. data/')
    parser.add_argument('--metafile', type=str, default='data/mimic/test.csv', help='path of meatafile')
    parser.add_argument('--batch', type=int, default=32, help='batch size')

    # save infos
    parser.add_argument('--save_path', type=str, default='results/for_test.csv', help='path to save frames')
        
    # dataparallel
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3,4,5,6], help='gpu to use')
    
    args = parser.parse_args()

    # run main
    main(args)