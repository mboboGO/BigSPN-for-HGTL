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

    ''' save path '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
    args.num_classes,args.sf_size = args.c_att.size()

    if len(range(torch.cuda.device_count()))>1:
                args.c_att = torch.cat([args.c_att,args.c_att],dim=0)
                args.f_att = torch.cat([args.f_att,args.f_att],dim=0)
    args.trans_map = torch.from_numpy(data_info['trans_map'][...]).cuda()


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
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(pse_optimizer, epoch)
        adjust_learning_rate(ste_optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, pse_optimizer, ste_optimizer, epoch,is_fix=args.is_fix)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        if args.is_fix:
            save_path = os.path.join(args.save_path,'fix.model')
        else:
            save_path = os.path.join(args.save_path,args.arch+('_{:.4f}.model').format(best_prec1))
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                #'optimizer' : optimizer.state_dict(),
            },filename=save_path)
            print('saving!!!!')

def train(train_loader, model, criterion, cls_optimizer, trans_optimizer, epoch,is_fix):    
    # switch to train mode
    model.train()
    if(is_fix):
        freeze_bn(model) 

    batch_start = time.time()
    for i, (input, target,_) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        start = time.time()
        logits,feats = model(input,args.c_att,args.f_att)
        inf_time = time.time()-start
        
        # compute gradient and do SGD step
        start = time.time()
        L_cls,L_sem,L_trans = criterion(target,logits)
        cls_optimizer.zero_grad()
        (L_cls+L_sem).backward(retain_graph=True)        
        cls_optimizer.step()
        trans_optimizer.zero_grad()
        (L_trans).backward()
        trans_optimizer.step()
        update_time = time.time()-start
        
        batch_time = time.time()-batch_start
        batch_start = time.time()    
        
        if i % args.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Inf Time {:.3f}\t'
                  'Update Time {:.3f}\t'
                  'Batch Time {:.3f}\t'.format(epoch,i,
                  len(train_loader),inf_time,update_time,batch_time))
            print('L_cls {:.4f} L_sem {:.4f} L_trans {:.4f}'.format(L_cls.item(),L_sem.item(),L_trans.item()))

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
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits[0], target_c, topk=(1, 5))
            bas_acc.update(prec1[0], input.size(0))
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits[2], target_f, topk=(1, 5))
            sub_acc1.update(prec1[0], input.size(0))
            # opt
            c_logit = logits[0].cpu().numpy();
            c_pre = np.argmax(c_logit, axis=1)
            f_logit = logits[2]*args.trans_map[c_pre,:].float()
            prec1, prec5 = accuracy(f_logit, target_f, topk=(1, 5))
            sub_acc2.update(prec1[0], input.size(0))
            sub_acc3.update(prec5[0], input.size(0))
       
    print(' * Basic ACC {:.3f} Sub ACC1@ {:.3f} ACC2@ {:.3f} ACC3@ {:.3f}'.format(bas_acc.avg,sub_acc1.avg,sub_acc2.avg,sub_acc3.avg))

    return sub_acc2.avg.cpu().numpy()
   
def adjust_learning_rate(optimizer , epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        

if __name__ == '__main__':
    main()
