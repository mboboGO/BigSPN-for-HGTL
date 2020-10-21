import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from  models.MPNCOV import MPNCOV
import torch.nn.functional as F
import models.resnet
import models.densenet
import models.senet
from models.operations import *

import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['as3']
       
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        self.inplanes = 64
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.is_att = True
        self.arch = args.backbone
        
        super(Model, self).__init__()

        ''' backbone net'''
        block = Bottleneck
        layers = [3, 4, 23, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
            
        feat_dim = 2048
        ''' Parent-grained Semantic Embedding '''
        self.pse_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pse_sem = nn.Sequential(
            nn.Linear(sf_size,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,feat_dim),
            nn.LeakyReLU(),
        )
        self.pse_cls = nn.Linear(feat_dim, num_classes)
        
        ''' Sub-grained Transfer Embedding'''
        cps_dim = int(feat_dim/2)
        self.ste_cps = nn.Sequential(
            nn.Conv2d(feat_dim, cps_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(cps_dim),
            nn.ReLU(inplace=True),
        )
        self.ste_spal =  nn.Sequential(
            nn.Conv2d(cps_dim, cps_dim, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(cps_dim, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.ste_spa2 =  nn.Sequential(
            nn.Conv2d(cps_dim, cps_dim, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(cps_dim, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.ste_spa3 =  nn.Sequential(
            nn.Conv2d(cps_dim, cps_dim, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(cps_dim, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.ste_spa4 =  nn.Sequential(
            nn.Conv2d(cps_dim, cps_dim, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(cps_dim, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ste_proj = nn.Sequential(
            nn.Linear(feat_dim,feat_dim),
            nn.LeakyReLU(),
        )
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, p_atr,s_atr):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)                                               
        x = self.layer4(x)
        last_conv = x
		
   
        ''' Parent-grained Semantic Embedding '''
        x_pse = self.pse_proj(last_conv).view(x.size(0),-1)
        sem_cls = self.pse_sem(p_atr)
        w_norm = F.normalize(sem_cls, p=2, dim=1)
        x_norm = F.normalize(x_pse, p=2, dim=1)
        logit_pse = x_norm.mm(w_norm.permute(1,0))
        logit_cls = self.pse_cls(x_pse)
   
        ''' Sub-grained Transfer Embedding '''
        x = self.ste_cps(last_conv)
        # gen att
        att1 = self.ste_spal(x) 
        att2 = self.ste_spa2(x)
        #att3 = self.ste_spa3(x)
        #att4 = self.ste_spa4(x)
        # att
        x_att1 = x*att1
        x_att2 = x*att2
        #x_att3 = x*att3
        #x_att4 = x*att4
        x = torch.cat([x_att1,x_att2],dim=1)
        x = self.avgpool(x).view(x.size(0),-1)
        x = self.ste_proj(x)
        # trans
        sem_cls = self.pse_sem(s_atr)
        w_norm = F.normalize(sem_cls, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        logit_ste = x_norm.mm(w_norm.permute(1,0))
        
        return (logit_cls,logit_pse,logit_ste),(last_conv)
		
		
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
		
        self.cls_loss = nn.CrossEntropyLoss()
        self.trans_map = args.trans_map
        
    def forward(self, label, logits):
        logit_cls = logits[0]
        logit_pse = logits[1]
        logit_ste = logits[2]
        
        ''' Loss '''
        # coarse
        L_cls = self.cls_loss(logit_cls,label)
        idx = torch.arange(logit_pse.size(0)).long()
        L_sem = (1-logit_pse[idx,label]).mean()
        
        # fine
        idx = self.trans_map[label,:].float()
        label_ = (logit_ste*idx).argmax(dim=1).long()
        num = torch.arange(logit_ste.size(0)).long()
        L_trans = (1-logit_ste[num,label_]).mean()
        
        return L_cls,L_sem,L_trans
		
def as3(pretrained=False, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained,args)
    loss_model = Loss(args)
    if pretrained:
        model_dict = model.state_dict()
        #pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('/model/mbobo/resnet101-5d3b4d8f/resnet101-5d3b4d8f.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,loss_model
	
	