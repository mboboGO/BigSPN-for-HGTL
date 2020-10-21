import numpy as np
import torch.nn as nn
#from sklearn.neighbors import KNeighborsClassifier


import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                 
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()

def preprocess_strategy(dataset):

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])  
    val_transforms = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms#, evaluate_transforms

        
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def adj_matrix(sf,gcn_k):
    '''adj'''
    norm = np.linalg.norm(sf,axis=1,keepdims=True)
    sf = sf/np.tile(norm,(1,sf.shape[1]))
    adj = np.dot(sf,sf.transpose(1,0))
    adj_sort = np.argsort(adj,axis=1)
    adj_sort = adj_sort[:,::-1]
    t = adj[np.arange(adj.shape[0]),adj_sort[:,gcn_k]]
    t = np.tile(t,(adj.shape[0],1)).transpose(1,0)
    idx = np.where(adj<t)
    adj[idx[0],idx[1]] = 0
    # norm
    rowsum = np.sum(adj,axis=1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    adj = r_mat_inv.dot(adj)
            
    return adj
                
                
                
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res