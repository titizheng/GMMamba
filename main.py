
import os


from utils.core import train_function,val_function 
from utils.createmode import create_model
from torch.utils.data import DataLoader
from datasets.toloda_datasets import h5file_Dataset
import torch.nn as nn
import torch
import numpy as np
import random
from datetime import datetime
from utils.utils import save_checkpoint

 
def seed_torch(seed=2021):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

def train_main(args):

    args.h5 = '/dataset/TCGA-ESCA/res18_features/h5_files'
    args.csv = f'/dataset/TCGA-ESCA/five_flod_csv/patient_labels_{args.flod}.csv'


    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.save_dir = os.path.join(args.save_dir,f'{args.arch}/{current_time}_{args.mask_ratio}/{args.flod}')
    os.makedirs(args.save_dir, exist_ok=True)  
    
    basedmodel,classifymodel,memory,super_token_model = create_model(args)


    train_dataset = h5file_Dataset(args.csv,args.h5,'train') 
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=2, pin_memory=True, shuffle=True) #
    valid_dataset = h5file_Dataset(args.csv,args.h5,'val')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2, pin_memory=True,  shuffle=True)

    test_dataset = h5file_Dataset(args.csv,args.h5,'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=True,shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classify_loss_fn = nn.CrossEntropyLoss().cuda()

    optimizer_classify_loss = torch.optim.AdamW(list(basedmodel.parameters())+list(classifymodel.parameters())+list(super_token_model.parameters()), lr=1e-4, weight_decay=1e-5)  

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0,'test_precision':0,'test_recall':0}
  
    patience = 20 
    counter = 0     


    for epoch in range(args.epoch): 
        train_auc, train_acc, train_f1, train_precision, train_recall =train_function(args,basedmodel,super_token_model,classifymodel, memory,
                                                                                      epoch,train_loader,device, classify_loss_fn, optimizer_classify_loss)
        
        valid_auc, valid_acc, valid_f1, valid_precision, valid_recall = val_function(args,basedmodel,super_token_model,classifymodel, memory,valid_loader,device,epoch)
        
        
        
        

        if   epoch > 5 and  valid_f1 + valid_auc > best_state['val_f1'] + best_state['val_auc']: #  
            test_auc, test_acc, test_f1, test_precision, test_recall = val_function(args,basedmodel,super_token_model,classifymodel, memory,test_loader,device,epoch,currenttype='test')
            counter = 0  
            best_state['epoch'] = epoch
            best_state['val_auc'] = valid_auc
            best_state['val_acc'] = valid_acc
            best_state['val_f1'] = valid_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            best_state['test_precision'] = test_precision
            best_state['test_recall'] = test_recall

    
            torch.save({
                'epoch': epoch ,
                'model_state_dict': basedmodel.state_dict(),
                'super_token_model':super_token_model.state_dict(),
                'fc': classifymodel.state_dict(),
            }, args.save_dir + f"/{epoch}_best_model.pth")
            print(f"🔥 Best Model Saved at Epoch {epoch}, AUC: {best_state['test_auc']:.4f}")
        else:
            counter += 1
            print(f"⚠️ No improvement. Early stop counter: {counter}/{patience}")
        
        # ---- Early Stopping ----
        if counter >= patience and epoch>50:
            print("⏹ Early stopping triggered!")
            break


    current_time = datetime.now().strftime("%Y-%m-%d")
    save_best_state = os.path.join(args.save_dir,f'output_file_{current_time}.txt') 
    with open(save_best_state, 'w') as file:
        for key, value in best_state.items():
            file.write(f'{key}: {value}\n')

    
        
        