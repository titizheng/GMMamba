import torch
import random
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #x = x.squeeze(dim=0)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1,attn_mode='normal'):
        super(AttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.mode = attn_mode
        self.attn = Attention(self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
    def forward(self,x):
        return x + self.attn(x)




class GroupsSuperAttenLayer(nn.Module): 
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.AttenLayer = AttenLayer(dim =self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
    def forward(self,data):

        class_token, super_features = data
        datas = torch.cat((class_token,super_features),dim=1)
        datas = self.AttenLayer(datas) 
        class_token = datas[:,0] 
        super_features = datas[:,1:]
        data = class_token, super_features
        return data


    
class ClassMultiMILTransformer(nn.Module):
    def __init__(self,args):
        super(ClassMultiMILTransformer, self).__init__()

        self.args = args
        self.fc2 = nn.Linear(self.args.embed_dim, self.args.n_classes)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.args.embed_dim))
    
        #--->build layers
        self.GroupsSuper_Layer = nn.ModuleList()
        for i_layer in range(self.args.num_layers):
            layer = GroupsSuperAttenLayer(dim=self.args.embed_dim)
            self.GroupsSuper_Layer.append(layer)

    def head(self,x):
        logits = self.fc2(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat} 
        return results_dict
    

    def forward(self,x,coords=False,mask_ratio=0):
 
        clstoken = self.cls_token 
        data = (clstoken, x) 
        for i in range(len(self.GroupsSuper_Layer)): 
            data = self.GroupsSuper_Layer[i](data)

        clstoken, x_groups = data 
        clstoken = clstoken.view(1,self.args.embed_dim)
        results_dict = self.head(clstoken) 
        
        return results_dict 
    


    
        
       

