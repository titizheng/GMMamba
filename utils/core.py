import os
import sys
sys.path.append(os.path.abspath('/GMMamba'))
from datasets.pre_input_data import grouping

import torchmetrics
import torch.nn as nn
import torch
import copy
from utils.utils import calculate_error,calculate_metrics_two,calculate_metrics_overtwo,f1_score,split_array,save_checkpoint,cosine_scheduler

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
# import tqdm
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import csv
import joblib

@torch.no_grad()
def val_function(args,basedmodel,super_token_model,classifymodel, memory,valid_loader,device,epoch,currenttype='valid'): 

    basedmodel.eval()
    classifymodel.eval()
    super_token_model.eval()

    
    valid_error = 0.
    Mamba_valid_error = 0.
    with torch.no_grad():
        val_label_list = []
        val_Y_prob_list = []
        Mamba_val_label_list = []
        Mamba_val_Y_prob_list = []
        for idx, (coords, data, label,data_dir) in enumerate (valid_loader):
            val_instancetoken = []
            val_confidence_last = 0.
            val_msg_logits_list = [] 

            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
   
            file_path = f'{data_dir[0]}_{args.num_subbags}.pkl' 
            if os.path.exists(file_path):
                model_data = joblib.load(file_path)
                features_group = model_data['features_group']
            else:
                grouping_instance = grouping(args.num_subbags) 
                features_group = grouping_instance.coords_grouping(update_coords,update_data)


            for patch_step in range(0, len(features_group)):
                h = basedmodel(features_group[patch_step],first=True)
                memory.msg_states.append(h)
            x = torch.stack(memory.msg_states[:], dim=1).view(1,-1,1,512) 
            group_super_features = super_token_model.stoken_forward(x)

            W_results_dict = classifymodel (group_super_features)  
            W_logits, W_Y_prob, W_Y_hat = W_results_dict['logits'], W_results_dict['Y_prob'], W_results_dict['Y_hat']

            
            

            error = calculate_error(W_Y_hat, label)
            valid_error += error

            memory.clear_memory()
            val_label_list.append(label)
            val_Y_prob_list.append(W_Y_prob)

            
           

        valid_acc = 1.0 - valid_error / len(valid_loader)


        targets = np.asarray(torch.cat(val_label_list, dim=0).cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(val_Y_prob_list, dim=0).cpu().numpy()) 
        
        if args.n_classes == 2:
            valid_auc, valid_acc, valid_f1, valid_precision, valid_recall = calculate_metrics_two(targets, probs)
        else:
            valid_auc, valid_acc, valid_f1, valid_precision, valid_recall = calculate_metrics_overtwo(targets, probs)

        print(f'{currenttype} Accuracy: {valid_acc:.4f} ," Precision: {valid_precision:.4f},Recall: {valid_recall:.4f},F1 Score: {valid_f1:.4f},AUC: {valid_auc:.4f}')

    save_result(args,epoch,currenttype,valid_acc,valid_auc,valid_f1,valid_precision,valid_recall)

    return valid_auc, valid_acc, valid_f1, valid_precision, valid_recall



def save_result(args,epoch,run_type,test_acc,test_auc,test_f1,test_precision,test_recall):
    current_time = datetime.now().strftime("%Y-%m-%d")
    save_csv_path = os.path.join(args.save_dir, f'metrics_{current_time}.csv')
    if not os.path.exists(save_csv_path):
        with open(save_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'type', 'acc', 'auc', 'f1', 'precision', 'recall'])

    with open(save_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,  
            run_type,  
            test_acc,  
            test_auc, 
            test_f1, 
            test_precision,  
            test_recall  
        ])
 



def train_function(args,basedmodel,super_token_model,classifymodel, memory,epoch,train_loader,
               device, classify_loss_fn, optimizer_classify_loss): 

    basedmodel.train()
    classifymodel.train()
    super_token_model.train()
    

    train_loss = 0.
    train_error = 0.
    train_bag_Y_prob = []
    train_bag_label = []
    
    progress_bar = tqdm(train_loader, desc="Training", position=0, leave=True)
   
    for idx, (coords, data, label,data_dir) in enumerate(progress_bar):
        update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long() 
        
        file_path = f'{data_dir[0]}_{args.num_subbags}.pkl' 
        if os.path.exists(file_path):
            model_data = joblib.load(file_path)
            features_group = model_data['features_group']
        else:
            grouping_instance = grouping(args.num_subbags) 
            features_group = grouping_instance.coords_grouping(update_coords,update_data)
        
        for patch_step in range(0, len(features_group)):
            h = basedmodel(features_group[patch_step],first=True)
            memory.msg_states.append(h)
        
        x = torch.stack(memory.msg_states[:], dim=1).view(1,-1,1,512) 
        group_super_features = super_token_model.stoken_forward(x)
        W_results_dict = classifymodel (group_super_features)
        W_logits, W_Y_prob, W_Y_hat = W_results_dict['logits'], W_results_dict['Y_prob'], W_results_dict['Y_hat']


        

        classify_loss = classify_loss_fn(W_logits, label) 
        train_loss += classify_loss.item()
        error = calculate_error(W_Y_hat, label)
        train_error += error


        optimizer_classify_loss.zero_grad()
        classify_loss.backward()
        optimizer_classify_loss.step()
        
        memory.clear_memory()

        progress_bar.set_description(f"Training (epoch = {epoch},loss={classify_loss.item():.4f}, error={error:.4f})")

        train_bag_Y_prob.append(W_Y_prob)
        train_bag_label.append(label)


    probs = np.asarray(torch.cat(train_bag_Y_prob, dim=0).detach().cpu().numpy())
    targets = np.asarray(torch.cat(train_bag_label, dim=0).cpu().numpy()).reshape(-1)
    
    if args.n_classes == 2:
        train_auc, train_acc, train_f1, train_precision, train_recall = calculate_metrics_two(targets, probs)
    else:
        train_auc, train_acc, train_f1, train_precision, train_recall = calculate_metrics_overtwo(targets, probs)

    train_error /= len(train_loader)
    train_loss /= len(train_loader)
    print('train_loss: {:.4f}'.format(train_loss))
    progress_bar.close()
    print(f'Epoch {epoch}: Train_Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
    return train_auc, train_acc, train_f1, train_precision, train_recall







    






