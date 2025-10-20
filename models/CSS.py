import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Memory:
    def __init__(self):

        self.msg_states = [] 

    def clear_memory(self):
        del self.msg_states[:] 


class CSS_block(nn.Module):
    def __init__(self, dim, n_classes, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.stoken_refine = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias)


        
    def stoken_forward(self, x):
        '''
           x: (B, N, 1, C) -> 

        '''
        B, N, _, C = x.shape
        x = x.squeeze(2)  
        ### MaxPooling
        max_super_feature, _ = x.max(dim=1, keepdim=True)  
        ### cross-attention
        affinity_matrix = torch.bmm(x, max_super_feature.transpose(1, 2))  
        affinity_matrix = F.softmax(affinity_matrix, dim=1)  
        cross_super_feature = torch.bmm(affinity_matrix.transpose(1, 2), x)  
        ### MHA
        refined_features, _ = self.stoken_refine(cross_super_feature.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        MHA_super_feature = refined_features.transpose(0, 1)  

        matrix_MHA_super_feature = torch.bmm(affinity_matrix, MHA_super_feature) 
        super_features = (matrix_MHA_super_feature + cross_super_feature).view(1,-1,512) 

        return super_features

        