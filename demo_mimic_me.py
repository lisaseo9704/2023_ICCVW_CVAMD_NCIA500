import argparse
import os
import torch
from torch.optim import Adam
import torch.nn.functional as F

from models.engine import *
from models.ml_gcn_me import *
from data.dataset_mimic import *
from utils.util import *
from models.loss2 import *


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', default='/data1/ICCV_data/resized_1024', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=1024, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0,1,2,3,4], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint/mimic_tr/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# sampling option
parser.add_argument('-cbs', '--cb_sampling', default=True,
                    help='True:class balanced sampling / False:random sampling')
parser.add_argument('-cb_opt', '--class_choice', default='cycle',
                    help='least_sampled, random, cycle')

# optimizer option(for Adam)
parser.add_argument('--G_optimizer_lr', '--opt_lr', default=1e-4, type=float,
                    metavar='W', help='G_optimizer_lr (default: 1e-4)')
parser.add_argument('--G_optimizer_betas', '--opt_beta', default=[0.9,0.99], type=float,
                    metavar='W', help='G_optimizer_betas (default: [0.9,0.99])')
parser.add_argument('--G_optimizer_wd', '--opt_wd', default=0, type=int,
                    metavar='W', help='G_optimizer_wd (default: 0)')

# optimizer option(for SGD)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def main_mimic():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    train_dataset = dataset_mimic_aug(args.data, phase='train', inp_name='data/mimic/mimic_glove_word2vec.pkl', metafile_path='data/mimic/train_train.csv')
    val_dataset = dataset_mimic(args.data, phase='val', inp_name='data/mimic/mimic_glove_word2vec.pkl', metafile_path='data/mimic/train_val.csv')
    num_classes = 26

    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/mimic/mimic_adj.pkl', block=Bottleneck)

    # define loss function (criterion)
    # criterion = nn.MultiLabelSoftMarginLoss(weight=None)
    criterion = RIDELoss(cls_num_list=None)

    # define optimizer
    # optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
    #                             lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = define_optimizer(model, args)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
            'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/mimic_tr/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device_ids'] = args.device_ids
    state['class_choice'] = args.class_choice

    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer, args.cb_sampling)

def define_optimizer(model, args):
    G_optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            G_optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    return Adam(G_optim_params, lr=args.G_optimizer_lr,
                            betas=args.G_optimizer_betas,
                            weight_decay=args.G_optimizer_wd)

if __name__ == '__main__':
    main_mimic()
