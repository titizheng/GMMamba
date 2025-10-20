
'''
Grouping WSI features preprocessed by CLAM based on location-based clustering
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append(os.path.abspath('/ICCV_GMMamba/'))

from utils.utils import make_parse
# from utils.core import save_kmeans_features  
from torch.utils.data import DataLoader
from datasets.toloda_datasets import h5file_Dataset
import torchmetrics
import torch.nn as nn
import torch
import copy

import numpy as np
from sklearn.cluster import KMeans

import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast

import joblib
from utils.utils import make_parse
# import os

class grouping:

    def __init__(self,groups_num):
        self.groups_num = groups_num
    
    def indicer(self, labels):
        indices = []
        groups_num = len(set(labels))
        for i in range(groups_num):
            temp = np.argwhere(labels==i).squeeze()
            indices.append(temp)
        return indices
    
    def make_subbags(self, idx, features):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0) 
            features_group.append(temp)
            
        return features_group
    
        
    def coords_nomlize(self, coords):
        coords = coords.squeeze()
        means = torch.mean(coords,0)
        xmean,ymean = means[0],means[1]
        stds = torch.std(coords,0)
        xstd,ystd = stds[0],stds[1]
        xcoords = (coords[:,0] - xmean)/xstd
        ycoords = (coords[:,1] - ymean)/ystd
        xcoords,ycoords = xcoords.view(xcoords.shape[0],1),ycoords.view(ycoords.shape[0],1)
        coords = torch.cat((xcoords,ycoords),dim=1)
        
        return coords
    
    
    def coords_grouping(self,coords,features,c_norm=False):
        features = features.squeeze()
        coords = coords.squeeze()
        # if c_norm:
        #     coords = self.coords_nomlize(coords.float())
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0, n_init='auto').fit(coords.cpu().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)
        
        return features_group
    
    def embedding_grouping(self,features):
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(features.cpu().detach().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)

        return features_group
    
    def random_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze() 
        indices = split_array(np.array(range(int(N))),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group
        
    def seqential_grouping(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = np.array_split(range(N),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group

    def idx_grouping(self,idx,features):
        idx = idx.cpu().numpy()
        idx = idx.reshape(-1)
        B, N, C = features.shape
        features = features.squeeze()
        indices = self.indicer(idx)
        features_group = self.make_subbags(indices,features)

        return features_group



def save_random_group(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):

            file_path = f'{data_dir[0]}_{args.num_subbags}_random.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags) 
            features_group = grouping_instance.random_grouping(update_data)
            model_data = {
            'features_group': features_group,  
            'names': f'{data_dir[0]}_{args.num_subbags}_random.pkl'              
            }

            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_random.pkl' )



def save_kmeans_features(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):

            file_path = f'{data_dir[0]}_{args.num_subbags}_featurekm.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags) 
            features_group = grouping_instance.embedding_grouping(update_data)
            model_data = {
            'features_group': features_group,  
            'names': f'{data_dir[0]}_{args.num_subbags}_featurekm.pkl'                    
            }
    
            
            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_featurekm.pkl' )


def save_seqential_grouping(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):
            
            file_path = f'{data_dir[0]}_{args.num_subbags}_coordsq.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags) 
            features_group = grouping_instance.seqential_grouping(update_data)
            model_data = {
            'features_group': features_group,  
            'names': f'{data_dir[0]}_{args.num_subbags}_coordsq.pkl'                    
            }
    
            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_coordsq.pkl')

def save_kmeans_coord(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):
            # instancetoken = []

            
            file_path = f'{data_dir[0]}_{args.num_subbags}_coordkm.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags) 
            features_group = grouping_instance.coords_grouping(update_coords,update_data)
            model_data = {
            'features_group': features_group,  
            'names': f'{data_dir[0]}_{args.num_subbags}_coordkm.pkl'             
            }
    
            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_coordkm.pkl')



def select_main(args):
    import pandas as pd
    import shutil
    args.num_subbags = 10

    flod = 'fold0'
    args.h5 = '/dataset/BRCA/feature_res18_level1_10x/h5_files'
    args.csv = f'/dataset/BRCA/flod_5_csv/patient_labels_{flod}.csv'


    test_dataset = h5file_Dataset(args.csv,args.h5,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    save_kmeans_coord(args,test_dataloader ) 


if __name__ == "__main__":

    args = make_parse()
    select_main(args)