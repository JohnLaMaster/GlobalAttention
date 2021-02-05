import torch.nn as nn
from collections import OrderedDict


__all__ = ['citation', 'GlobalAttention1d', 'GlobalAttention2d', 'GlobalAttention3d', 'freeze_global_attn']


citation = OrderedDict({'Title': 'A Novel Global Spatial Attention Mechanism in Convolutional Neual Network for Medical Image Classification',
                        'Authors': 'Linchuan Xu, Jun Huang, Atsushi Nitanda, Ryo Asaoka, and Kenji Yamanishi',
                        'Journal': 'Preprint',
                        'Institution': 'The Hong Kong Polytechnic University and The University of Tokyo',
                        'URL': 'https://arxiv.org/pdf/2007.15897',
                        'Notes': 'Code implementation modified from their original pixel CNN. Includes a function that '
                                 'can be called from the training script to freeze the weights.',
                        'Source Code': 'Ing. John T. LaMaster, September 2020'})


'''
Wriiten by: Ing. John T. LaMaster Oct. 2020

Inputs:
    num_channel:  number of input channels
                  This is used to keep the channels separate. If this is not desirable, set equal to zero.
    x:  input to model

Output:
    out:  input multiplied by the highlighting mask
    
freeze_global_attention can be called to freeze the parameters of the attention module as was done in 
the original publication.
'''


class GlobalAttention1d(nn.Module):
    def __init__(self, num_channel: torch.Int):
        super(GlobalAttention1d, self).__init__()
        self.conv1 = nn.Conv1d(num_channel, num_channel, 3, stride=1, padding=1, groups=num_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(num_channel, num_channel, 1, stride=1, padding=0, groups=num_channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor): -> torch.Tensor
        out = self.conv1(x)
        out = self.relu1(out)     
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return x.bmm(out)

class GlobalAttention2d(nn.Module):
    def __init__(self, num_channel: torch.Int):
        super(GlobalAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, 3, stride=1, padding=1, groups=num_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(num_channel, num_channel, 1, stride=1, padding=0, groups=num_channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor): -> torch.Tensor
        out = self.conv1(x)
        out = self.relu1(out)     
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return x.bmm(out)

class GlobalAttention3d(nn.Module):
    def __init__(self, num_channel: torch.Int):
        super(GlobalAttention3d, self).__init__()
        self.conv1 = nn.Conv3d(num_channel, num_channel, 3, stride=1, padding=1, groups=num_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(num_channel, num_channel, 1, stride=1, padding=0, groups=num_channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor): -> torch.Tensor
        out = self.conv1(x)
        out = self.relu1(out)     
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return x.bmm(out)

def freeze_global_attn(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, GlobalAttention1d) or \
               isinstance(child, GlobalAttention2d) or \
               isinstance(child, GlobalAttention3d):
                for param in child.parameters():
                    param.requires_grad = False
        else:
            freeze_global_attn(child)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_num_gen(gen):
    return sum(1 for x in gen)
    
