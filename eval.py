#from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
import torchvision.transforms as transforms
import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--backbone', default='resnet18', help='backbone')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')

                    
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)


    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:',args.seed)
    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    ''' data load info '''
    data_info = h5py.File(os.path.join('./data',args.data,'data_info.h5'), 'r')
    img_path = str(data_info['img_path'][...]).replace("b'",'').replace("'",'')
    args.c_att = torch.from_numpy(data_info['coarse_att'][...]).cuda()
    args.f_att = torch.from_numpy(data_info['fine_att'][...]).cuda()
    args.trans_map = torch.from_numpy(data_info['trans_map'][...]).cuda()
    args.num_classes,args.sf_size = args.c_att.size()

    ''' model building '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        best_prec1=0
        model,criterion = models.__dict__[args.arch](pretrained=True,args=args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](args=args)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda(args.gpu)
    
    ''' optimizer '''
    pse_params = [v for k, v in model.named_parameters() if 'ste' not in k]
    ste_params = [v for k, v in model.named_parameters() if 'ste' in k]
    
    pse_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pse_params), args.lr,
                                betas=(0.5,0.999),weight_decay=args.weight_decay)
    ste_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ste_params), args.lr,
                                betas=(0.5,0.999),weight_decay=args.weight_decay)

    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('./data',args.data,'train.list')
    valdir = os.path.join('./data',args.data,'test.list')

    train_transforms, val_transforms = preprocess_strategy(args.data)

    train_dataset = datasets.ImageFolder(img_path,traindir,train_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        


        
    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)

      

def validate(val_loader, model, criterion):
    bas_acc = AverageMeter()
    sub_acc1 = AverageMeter()
    sub_acc2 = AverageMeter()
    sub_acc3 = AverageMeter()
    val_corrects = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target_c,target_f) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target_c = target_c.cuda(args.gpu, non_blocking=True)
            target_f = target_f.cuda(args.gpu, non_blocking=True)
            
            # inference
            logits,feats = model(input,args.c_att,args.f_att)
            
            att1_ = feats[0].cpu().numpy()
            att2_ = feats[1].cpu().numpy()
            att3_ = feats[2].cpu().numpy()
            att4_ = feats[3].cpu().numpy()
            x1_ = feats[4].cpu().numpy()
            x2_ = feats[5].cpu().numpy()
            
            
            # evaluation
            if(i==0):
                gt_b = target_c.cpu().numpy()
                gt_s = target_f.cpu().numpy()
                att1 = att1_
                att2 = att2_
                att3 = att3_
                att4 = att4_
                x1 = x1_
                x2 = x2_
            else:
                gt_b = np.hstack([gt_b,target_c.cpu().numpy()])
                gt_s = np.hstack([gt_s,target_f.cpu().numpy()])
                att1 = np.vstack([att1,att1_])
                att2 = np.vstack([att2,att2_])
                att3 = np.vstack([att3,att3_])
                att4 = np.vstack([att4,att4_])
                x1 = np.vstack([x1,x1_])
                x2 = np.vstack([x2,x2_])
            
            
    ''' save feats '''
    f = h5py.File('data.h5', 'w')
    f.create_dataset('gt_b', gt_b.shape,dtype=gt_b.dtype)
    f.create_dataset('gt_s', gt_s.shape,dtype=gt_s.dtype)
    f.create_dataset('att1', att1.shape,dtype=att1.dtype)
    f.create_dataset('att2', att2.shape,dtype=att2.dtype)
    f.create_dataset('att3', att3.shape,dtype=att3.dtype)
    f.create_dataset('att4', att4.shape,dtype=att4.dtype)
    f.create_dataset('x1', x1.shape,dtype=x1.dtype)
    f.create_dataset('x2', x2.shape,dtype=x2.dtype)
    f['gt_b'][...] = gt_b
    f['gt_s'][...] = gt_s
    f['att1'][...] = att1
    f['att2'][...] = att2
    f['att3'][...] = att3
    f['att4'][...] = att4
    f['x1'][...] = x1
    f['x2'][...] = x2
    f.close()

    aaaa
   
def adjust_learning_rate(optimizer , epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        

if __name__ == '__main__':
    main()
