"""
intra-group masking Mamba (IMM)
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F




def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class IMM_block(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2, mask_ratio=0.2):
        super(IMM_block, self).__init__()
        self.in_dim = in_dim
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.layers = nn.ModuleList()
        self.survival = survival
        self.n_classes = n_classes
        
        self.norm = nn.LayerNorm(512)


        self.BiMamba1 = nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ))
        self.mask_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.BiMamba2 = nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ))

        self.group_ins_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.mask_ratio = mask_ratio
        # self.type = type

        self.apply(initialize_weights)

    def forward(self, x,first=False):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h_orgin = x.float()  # [B, n, 1024]
        len_ins = h_orgin.shape[1]

        h_0 = self.BiMamba1[0](h_orgin)
        h_1 = self.BiMamba1[1](h_0)
        BiMamba1_h_orgin = h_1 + h_orgin

        #-Group-masking 
        BiMamba1_h = self.norm(BiMamba1_h_orgin) 
        A = self.mask_attention(BiMamba1_h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        instance_score = A.view(-1)
        sorted_scores, sorted_indices = torch.sort(instance_score, descending=True)
        bottom_20_percent = int(self.mask_ratio * len_ins) #0.2
        mask = torch.ones(len_ins, dtype=torch.bool)  # True
        mask[sorted_indices[-max(1, bottom_20_percent):]] = False
        filtered_h = BiMamba1_h_orgin[:, mask, :]  


        filtered_h_0 = self.BiMamba2[0](filtered_h)
        filtered_h_1 = self.BiMamba2[1](filtered_h_0)
        BiMamba2_h = filtered_h_1 + filtered_h

        BiMamba2_h = self.norm(BiMamba2_h)
        A = self.group_ins_attention(BiMamba2_h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        BiMamba2_h = torch.bmm(A, BiMamba2_h) # [B, K, 512]

        return BiMamba2_h #

        
