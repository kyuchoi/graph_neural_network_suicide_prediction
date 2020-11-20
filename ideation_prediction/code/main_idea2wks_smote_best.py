# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:19:27 2020

ref: https://www.dgl.ai/blog/2019/01/25/batch.html

@author: CNDLMembers
"""

import os
import glob
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from utils_idea2wks_smote_best import *
from sklearn.metrics import confusion_matrix
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import natsort

parser = argparse.ArgumentParser()

# training
parser.add_argument('--model_arch', type=int, default=2, help='0 for GCN, 1 for GAT, and 2 for GIN')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--ncpu', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=35)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_best_model', action="store_true", default=False, help='use save_best_model function with various metrics')
parser.add_argument('--save_model_metrics', type=str, default='loss', help='choose among loss/sens/spec/acc')
parser.add_argument('--label_name', type=str, default='suicidal_idea_within_2wk') # 'suicidal_idea'

# directory
parser.add_argument('--new_subsample', action="store_true", default=False, help='subsample new balanced datasets to train')
parser.add_argument('--sample_opt', type=str, default='u2') # undersample: u1, u2 / oversample: SMOTE
parser.add_argument('--balance_ratio', type=int, default=10) # balance_ratio makes balance of pos:neg = 1:3
parser.add_argument('--split_ratio', type=float, default=0.1) # split_ratio makes train:valid = 9:1
parser.add_argument('--thr', type=str, default='6') # string for diff thr from R: 4,5,6 for thr 0.4, 0.5, 0.6
parser.add_argument('--raw_data_dir', type=str, default='../raw_data')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_dir', type=str, default='../log')
parser.add_argument('--ckpt_dir', type=str, default='../checkpoint')
parser.add_argument('--result_dir', type=str, default='../result')

# GIN args
parser.add_argument('--num_layers', type=int, default=6) # 5
parser.add_argument('--num_mlp_layers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=128) # 64
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--learn_eps', action="store_true")
parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "mean", "max"])
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "mean", "max"])

args = parser.parse_args()

batch_size = args.batch_size
seed = args.seed
lr = args.lr
ncpu = args.ncpu
num_epochs = args.num_epochs
args.best_epoch = 0

# set this for True for working
args.new_subsample = args.new_subsample # True # 
args.smote_ratio = 1.0 # 0.7

### set directory using thr
args.data_dir += "_thr" + args.thr
args.log_dir += "_thr" + args.thr
args.ckpt_dir += "_thr" + args.thr
args.result_dir += "_thr" + args.thr

# making seperate log and checkpoint directory for each setting: e.g. save_model_metrics
args.log_dir = os.path.join(args.log_dir, args.save_model_metrics, args.sample_opt)
log_dir = args.log_dir

# make dir for each sample option: e.g. SMOTE, u1, u2
args.ckpt_dir = os.path.join(args.ckpt_dir, args.save_model_metrics, args.sample_opt)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

args.result_dir = os.path.join(args.result_dir, args.sample_opt)
result_dir = args.result_dir

# log_dir = os.path.join(args.log_dir, args.save_model_metrics)
# args.ckpt_dir = os.path.join(args.ckpt_dir, args.save_model_metrics)
# result_dir = args.result_dir

raw_data_dir = args.raw_data_dir
args.data_dir = os.path.join(args.data_dir, f'{args.sample_opt}')

# define train, valid, and test directory
train_data_dir = os.path.join(args.data_dir, 'train')
valid_data_dir = os.path.join(args.data_dir, 'valid')
test_data_dir = os.path.join(args.data_dir, 'test')
csv_dir = os.path.join(raw_data_dir, 'data_for_smote')

# # set train_valid data directory    
# if not os.path.exists(train_data_dir):    
#     os.makedirs(train_data_dir)    
    
# if not os.path.exists(valid_data_dir):
#     os.makedirs(valid_data_dir)  

# you can just save constants without add_argument
args.csv_dir = csv_dir
args.ncls = 1 
epsilon = 1e-8

# set the metrics to save the best model
if args.save_model_metrics == 'loss':
    args.best_metrics = 1e10
else:
    args.best_metrics = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# make a checkpoint directory
# if os.path.exists(args.ckpt_dir):
#     shutil.rmtree(args.ckpt_dir)
# os.makedirs(args.ckpt_dir)

# make a result directory
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# make a log directory for tensorboard
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
os.makedirs(os.path.join(log_dir,'train'))
os.makedirs(os.path.join(log_dir,'valid'))

# fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
writer_loss = SummaryWriter(log_dir=os.path.join(log_dir, 'loss'))
writer_metrics = SummaryWriter(log_dir=os.path.join(log_dir, 'metrics'))

#%% Create dataset from csv files, and split into training, validation and test set. 

# set questionnaires, labels and single items
questions = {'PHQ':9,'GAD':7}
single_items=['Gender', 'site', 'STAI_X1_total', 'RAS_total', 'RSES_total', 'MaDE', 'suicidal_attempt']
label_lst=['suicidal_idea_within_2wk','ID']  # MUST put ID in the last for SMOTE !!

args.n_feats = 5 if 'RAS' in questions.keys() else 4 # only RAS got 5-scale, and the others have 4-scale # it's not the same as n_nodes, like 100 x 100 YAD-HCP case !!
lst_num_questions = [value for value in questions.values()] 
args.n_nodes = sum(lst_num_questions) + len(single_items)
# get feature names only (except labels)
args.lst_feature = get_feature_columns(questions, single_items, label_lst)[:args.n_nodes]

### make datasets into train, valid, and test directories

# set filename for train, valid, and test set here
lst_train_valid_filename = ['Unlabelled_KAIST.csv','Labelled_SMC.csv', 'Labelled_Gachon.csv', 'Labelled_KAIST.csv'] 
lst_test_filename = ['Unlabelled_SNU.csv'] 
lst_total_filename = lst_train_valid_filename + lst_test_filename

if args.new_subsample:
    
    use_split_into_single_csv = True # False #

    df_total = make_datasets(args, questions, single_items, label_lst, lst_total_filename).df_total_drop
    df_total.to_csv(os.path.join(args.csv_dir, 'df_total_drop.csv'), index=False)
    
    # reset train_valid data directory    
    if os.path.exists(train_data_dir):
        shutil.rmtree(train_data_dir)
    os.makedirs(train_data_dir)    
        
    if os.path.exists(valid_data_dir):
        shutil.rmtree(valid_data_dir)
    os.makedirs(valid_data_dir)    
    
    df_train_valid = make_datasets(args, questions, single_items, label_lst, lst_train_valid_filename).df_total_drop
    df_train_valid.to_csv(os.path.join(args.csv_dir, 'df_train_valid.csv'), index=False)
        
    df_train_valid_feat = df_train_valid.drop(args.label_name, axis=1)
    df_train_valid_label = df_train_valid[args.label_name]
    
    X = df_train_valid_feat.to_numpy(dtype=int) # losing all the column names here
    y = df_train_valid_label.to_numpy(dtype=int)
    
    # choose sampling method
    '''
    split into single csv files AFTER sampling here: over, under, and SMOTE-NC sampling    
    '''
    if args.sample_opt == 'SMOTE':
        print('sampling using SMOTE')
        categorical_feature_num = np.array([0,1]+list(range(4, X.shape[1]-1))) # except ID in the last !!
        smote_nc = SMOTENC(categorical_features=categorical_feature_num, random_state=args.seed, sampling_strategy=args.smote_ratio)
        X, y = smote_nc.fit_resample(X, y)
    elif args.sample_opt == 'u1':
        print(f'sampling using undersample with ratio of {args.balance_ratio}')
        X, y = under_sample(X, y, args.balance_ratio)
    elif args.sample_opt == 'u2':
        print(f'sampling using undersample with ratio of {args.balance_ratio//2}')
        X, y = under_sample(X, y, args.balance_ratio//2)
        
    ##### original code
    # use_SMOTE = True # False
    # if use_SMOTE:
                
    #     '''
    #     split into single csv files AFTER sampling here: over, under, and SMOTE-NC sampling    
    #     '''
    #     print('sampling using SMOTE')
    #     categorical_feature_num = np.array([0,1]+list(range(4, X.shape[1]-1))) # except ID in the last !!
    #     smote_nc = SMOTENC(categorical_features=categorical_feature_num, random_state=args.seed, sampling_strategy=args.smote_ratio)
    #     X, y = smote_nc.fit_resample(X, y)
    
    # split into train and valid dataset    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.split_ratio, random_state=args.seed, stratify=y)    
    
    X_train = pd.DataFrame(X_train, columns=df_train_valid_feat.columns)
    X_valid = pd.DataFrame(X_valid, columns=df_train_valid_feat.columns)
    y_train = pd.DataFrame(y_train, columns=[args.label_name])
    y_valid = pd.DataFrame(y_valid, columns=[args.label_name])
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_valid = pd.concat([X_valid, y_valid], axis=1)
    
    df_train = convert_total_quantile(args, df_train, dataset='train')
    df_valid = convert_total_quantile(args, df_valid, dataset='valid')
    
    df_train.to_csv(os.path.join(args.csv_dir, 'df_train.csv'), index=False)
    df_valid.to_csv(os.path.join(args.csv_dir, 'df_valid.csv'), index=False)
        
    if use_split_into_single_csv:
        split_into_single_csv(df_train, train_data_dir)
        split_into_single_csv(df_valid, valid_data_dir)
    else:
        print("not choose to split into single csv files")        

else:
    print('data already exists!')
    
    # # set test data directory
    # test_data_dir = os.path.join(args.data_dir, 'test')
    
    # if os.path.exists(test_data_dir):
    #     shutil.rmtree(test_data_dir)
    # os.makedirs(test_data_dir)   
    
    # df_test = make_datasets(args, questions, single_items, label_lst, lst_test_filename).df_total_drop
    # df_test = convert_total_quantile(df_test)
    # df_test.to_csv(os.path.join(args.csv_dir, 'df_test.csv'), index=False)
    
    # if use_split_into_single_csv:
    #     split_into_single_csv(df_test, test_data_dir)
    # else:
    #     print("not choose to split into single csv files")        

#%%
# set dataset and dataloader
dataset = {}
dataset['train'] = survey_dataset(args, train_data_dir)
dataset['valid'] = survey_dataset(args, valid_data_dir)
# dataset['test'] = survey_dataset(args, test_data_dir)

dataloader = {x: DataLoader(dataset=dataset[x], batch_size = batch_size if x != 'test' else 640, shuffle=True if (x =='train' or x == 'valid') else False, num_workers=ncpu, collate_fn = collate, pin_memory = True) for x in ['train', 'valid']} # , 'test'

# dataloader = {x: DataLoader(dataset=dataset[x], batch_size = batch_size if x != 'test' else 640, shuffle=True if (x =='train' or x == 'valid') else False, num_workers=ncpu, collate_fn = collate, pin_memory = True) for x in ['train', 'valid', 'test']} 

# get test graph drawn
g_train = dataset['train'].__getitem__(0)[0]
nx.draw_circular(g_train.to_networkx(), with_labels = True, node_size = args.n_nodes)

#%% define model with chosen architecture

model = build_model(args)
model = model.to(device)

# for class weighted loss
weights = [3.0] # float(args.balance_ratio-2)
class_weights = torch.FloatTensor(weights).to(device)

# define loss, optimizer, and scheduler
loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#%% train and valid

# setting the txt file to save the valid metrics in the result directory
with open(os.path.join(args.result_dir, 'val_metrics.txt'), 'w') as f:
    f.write('epoch, loss, sens, spec, acc\n')

# setting the directory to save tsne for each fold
tsne_path = os.path.join(args.result_dir, 'tsne_plots')
if os.path.exists(tsne_path):
    shutil.rmtree(tsne_path)
os.makedirs(tsne_path)

# setting the directory to save roc for each fold
roc_path = os.path.join(args.result_dir, 'roc_plots')
if os.path.exists(roc_path):
    shutil.rmtree(roc_path)
os.makedirs(roc_path)

epoch_losses = []
epoch_losses_val=[]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total = 0
    correct = 0
    tn, fp, fn, tp = 0, 0, 0, 0
    
    for iter, (bg, label) in enumerate(dataloader['train']):
        
        feats = bg.ndata.pop('n')
        feats, label = feats.to(device), label.to(device)
        
        prediction = model(bg, feats)
    
        loss = loss_func(prediction, label) 
        
        prediction = torch.sigmoid(prediction)
        # pred_score, pred_class = torch.max(prediction, dim=1)
        pred_class = (prediction>0.5).float()
        
        total += label.size(0)
        correct += torch.sum(pred_class == label).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader['train']) # same as (iter+1)
        
        # get metrics
        tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
    
        tn+=tn_batch
        fp+=fp_batch
        fn+=fn_batch
        tp+=tp_batch        

    acc = correct/total * 100
        
    # print metrics
    sens = tp/(tp+fn+epsilon) * 100
    spec = tn/(tn+fp+epsilon) * 100
    new_acc = (tp+tn)/(tn+fp+fn+tp+epsilon) * 100        
    
    print('Epoch {}/{}, loss {:.4f}, sens {:.4f}, spec {:.4f}, acc {:.4f}, new_acc {:.4f}'.format(epoch+1, num_epochs, epoch_loss, sens, spec, acc, new_acc))
    epoch_losses.append(epoch_loss)
    
    writer_loss.add_scalars('loss/train', {'loss': np.mean(epoch_losses)}, epoch)
    writer_metrics.add_scalars('metrics/train', {'sens': sens,
                                                 'spec': spec,
                                                 'acc': acc
                                                 }, epoch)
    scheduler.step()
    
    # valid for only once at the last
    with torch.no_grad():
        model.eval()
        epoch_loss_val = 0
        val_total = 0
        val_correct = 0
        
        lst_final_concat = []
        lst_labels = []
        lst_prediction = []
        lst_pred_scores = []
        lst_pred_class = []
        
        tn, fp, fn, tp = 0, 0, 0, 0
        
        for batch, (bg, label) in enumerate(dataloader['valid']):
            
            feats = bg.ndata.pop('n')
            feats, label = feats.to(device), label.to(device)
            
            prediction = model(bg, feats)
            final_concat = model.final_concat
            
            prediction = torch.sigmoid(prediction)
            # pred_score, pred_class = torch.max(prediction, dim=1)
            pred_class = (prediction>0.5).float()
            
            final_concat_np = fn_tonumpy(final_concat)
            pred_class_np = fn_tonumpy(pred_class)
            # pred_score_np = fn_tonumpy(pred_score)
            prediction_np = fn_tonumpy(prediction)
            labels_np = fn_tonumpy(label)
            
            lst_final_concat.extend(final_concat_np)
            lst_pred_class.extend(pred_class_np)
            lst_labels.extend(labels_np)
            lst_prediction.extend(prediction_np)
            # lst_pred_scores.extend(pred_score_np)
            
            # get metrics                
            tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
    
            tn+=tn_batch
            fp+=fp_batch
            fn+=fn_batch
            tp+=tp_batch
                            
            val_loss = loss_func(prediction, label) 
            epoch_loss_val += val_loss.detach().item()
            epoch_loss_val /= len(dataloader['valid'])
            
        # print metrics
        val_sens = tp/(tp+fn+epsilon) * 100
        val_spec = tn/(tn+fp+epsilon) * 100
        val_acc = (tp+tn)/(tn+fp+fn+tp+epsilon) * 100        
        epoch_losses_val.append(epoch_loss_val)
        
        print(f'Epoch {epoch+1}/{num_epochs}: valid loss {np.mean(epoch_losses_val):.4f}, valid sens {val_sens:.2f}, valid spec {val_spec:.2f}, valid acc {val_acc:.2f}')
        
        with open(os.path.join(args.result_dir, 'val_metrics.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}: {epoch_loss_val:.4f}, {val_sens:.2f}, {val_spec:.2f}, {val_acc:.2f}\n')
        
        # saving valid labels and pred_classes for AUC
        print(f'Saving validation labels and pred_classes: ')
        val_label_pred_path = os.path.join(args.result_dir, 'val_label_pred', f'epoch{epoch+1}')
        if os.path.exists(val_label_pred_path):
            shutil.rmtree(val_label_pred_path)
        os.makedirs(val_label_pred_path)
        np.savetxt(os.path.join(val_label_pred_path, f'val_labels_e{epoch+1}.txt'), lst_labels, fmt = '%d', delimiter = ',')
        np.savetxt(os.path.join(val_label_pred_path, f'val_pred_classes_e{epoch+1}.txt'), lst_pred_class, fmt = '%d', delimiter = ',')
        np.savetxt(os.path.join(val_label_pred_path, f'val_predictions_e{epoch+1}.txt'), lst_prediction, fmt = '%f', delimiter = ',')
        
        writer_loss.add_scalars('loss/valid', {'loss': np.mean(epoch_losses_val)}, epoch)
        writer_metrics.add_scalars('metrics/valid', {'sens': val_sens,
                                                     'spec': val_spec,
                                                     'acc': val_acc
                                                     }, epoch)
        
        save(args, model, optimizer, epoch)
        
        if True: # epoch == num_epochs - 1:
            # draw t-sne for each fold
            plt.ioff()
            plt.figure(figsize=(15,10))
            plt.title(f"plotting t-SNE for {args.label_name}")
            draw_tsne_plot(args, lst_final_concat, lst_labels, '{args.label_name}') # lst_prediction                    
            print('Saving tsne plot for valid')
            plt.savefig(os.path.join(tsne_path, f'tsne_plot_for_valid_e{epoch+1}.png'), dpi=300)
            plt.close()
            
            ### get ROC and AUC for each fold
            lst_labels = np.array(lst_labels, dtype=int)
            lst_prediction = np.array(lst_prediction, dtype=float)
            
            fpr, tpr, _ = roc_curve(lst_labels, lst_prediction) 
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC curve for valid set (AUC = {roc_auc:0.2f})') # alpha: transparency of curves
            
            # for saving plots for each fold
            plt.ioff()                
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label=f'ROC curve for valid set (AUC = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for valid set')
            plt.legend(loc="lower right")
            print('Saving ROC plot for valid set')
            plt.savefig(os.path.join(roc_path, f'roc_plot_for_valid_set_e{epoch+1}.png'), dpi=300)
            # plt.show()
            plt.close()
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
                
    writer_loss.close()    
    writer_metrics.close()
