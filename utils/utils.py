import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch

import os
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize



def make_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', default='TCGA',type=str) #camelyon16 TCGA
    # parser.add_argument('--mode', default='rlselect',type=str)
    parser.add_argument('--seed', default=1,type=int)
    parser.add_argument('--epoch', default=100,type=int)
    parser.add_argument('--lr', default=0.00001,type=int)
    
    #basedmodel
    parser.add_argument('--in_chans', default=512,type=int)
    parser.add_argument('--embed_dim', default=512,type=int)
    parser.add_argument('--num_subbags', default=10,type=int) 
    parser.add_argument('--mask_ratio', default=0.3,type=float,help='Masking Ratio,0.2,0.3,0.4,0.5')  


    
    parser.add_argument('--attn', default='normal',type=str)
    parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--num_layers', default=1,type=int)  
    

    parser.add_argument('--split_group', default='region_group',type=str,choices=['instance_seuqence','coords_kmenas','region_group']) 

    parser.add_argument('--mambamil_layer',type=int, default=1, help='mambamil_layer')
    parser.add_argument('--mambamil_type',type=str, default='BiMamba', choices= ['Mamba', 'BiMamba'], help='mambamil_type')
    parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')

    parser.add_argument('--save_dir', default='/result',type=str)
    parser.add_argument('--train_h5',default='',type=str)
    parser.add_argument('--csv', default='',type=str)

    
    args = parser.parse_args()
    return args



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    schedule_per_epoch = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            value = np.linspace(start_warmup_value, base_value, warmup_epochs)[epoch]
        else:
            iters_passed = epoch * niter_per_ep
            iters_left = epochs * niter_per_ep - iters_passed
            alpha = 0.5 * (1 + np.cos(np.pi * iters_passed / (epochs * niter_per_ep)))
            value = final_value + (base_value - final_value) * alpha
        schedule_per_epoch.append(value)
    return schedule_per_epoch




def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error



def calculate_metrics_two(y_true, probs):
    pos_probs = probs[:, 1] if probs.ndim > 1 else probs
    y_pred = (pos_probs >= 0.5).astype(int)
    test_precision = precision_score(y_true, y_pred)  
    test_recall = recall_score(y_true, y_pred)  
    test_f1 = f1_score(y_true, y_pred)  
    test_auc = roc_auc_score(y_true, pos_probs)
    test_acc = accuracy_score(y_true, y_pred)
    return test_auc, test_acc, test_f1, test_precision, test_recall



def calculate_metrics_overtwo(y_true, probs):
    y_pred = np.argmax(probs, axis=1) 
    test_precision = precision_score(y_true, y_pred, average='macro') 
    test_recall = recall_score(y_true, y_pred, average='macro') 
    test_f1 = f1_score(y_true, y_pred, average='macro') 
    y_true_one_hot = label_binarize(y_true, classes=[0, 1, 2])
    test_auc = roc_auc_score(y_true_one_hot, probs, average='macro')
    test_acc = np.mean(y_pred == y_true)
    return test_auc, test_acc, test_f1, test_precision, test_recall



def coords_nomlize(coords):
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




def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False) 
    split_indices = np.array_split(indices, m)  

    result = []
    for indices in split_indices:
        result.append(array[indices])

    return result




def early_stopping(epoch, val_loss, model, args, 
                   patience=20, stop_epoch=50, 
                   ckpt_name='', best_state=None, 
                   counter_state=None):
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    Args:
        epoch (int):  epoch
        val_loss (float):  loss
        model (nn.Module): 
        args: 
        patience (int):  epoch 
        stop_epoch (int): epoch
        ckpt_name (str): 
        best_state (dict):  { "best_score": float, "val_loss_min": float }
        counter_state (dict):  counter { "counter": int }
        
    Return:
        early_stop (bool), best_state, counter_state
    """

    if best_state is None:
        best_state = {"best_score": None, "val_loss_min": np.Inf}
    if counter_state is None:
        counter_state = {"counter": 0}

    ckpt_name = ckpt_name if ckpt_name else f'./ckp/{args.type}_checkpoint_{args.seed}_{epoch}.pt'
    score = -val_loss
    early_stop = False


    if best_state["best_score"] is None:
        best_state["best_score"] = score
        best_state["val_loss_min"] = val_loss
        torch.save(model.state_dict(), ckpt_name)
        print(f"✅ Model saved at epoch {epoch}, val_loss: {val_loss:.4f}")
    elif score < best_state["best_score"]:
        counter_state["counter"] += 1
        print(f"⚠️ EarlyStopping counter: {counter_state['counter']} out of {patience}")
        if counter_state["counter"] >= patience and epoch > stop_epoch:
            early_stop = True
    else:
        best_state["best_score"] = score
        best_state["val_loss_min"] = val_loss
        torch.save(model.state_dict(), ckpt_name)
        print(f"✅ Model saved at epoch {epoch}, val_loss: {val_loss:.4f}")
        counter_state["counter"] = 0

    return early_stop, best_state, counter_state

def save_checkpoint(state,best_acc, auc,epoch,checkpoint, filename='checkpoint.pth.tar'):
    best_acc = f"{best_acc:.4f}"
    auc = f"{auc:.4f}"
    filepath = os.path.join(checkpoint, str(epoch)+"_"+best_acc+"_"+auc+"_"+filename)
    torch.save(state, filepath)



