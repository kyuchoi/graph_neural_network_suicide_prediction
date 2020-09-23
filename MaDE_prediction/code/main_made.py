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
from utils_made import *
import argparse
import matplotlib.pyplot as plt

    
parser = argparse.ArgumentParser()
# training
parser.add_argument('--model_arch', type=int, default=2, help='0 for GCN, 1 for GAT, and 2 for GIN')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--seed', type=int, default=9877)
parser.add_argument('--ncpu', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_best_model', action="store_true", default=False, help='use save_best_model function with various metrics')
parser.add_argument('--save_model_metrics', type=str, default='loss', help='choose among loss/sens/spec/acc')
parser.add_argument('--ext_center', type=str, default='Gachon') # SMC # Gachon # KAIST
parser.add_argument('--get_saliency', action="store_true", default=False, help='get saliency for test set')

# directory
parser.add_argument('--new_subsample', action="store_true", default=False, help='subsample new balanced datasets to train')
parser.add_argument('--raw_data_dir', type=str, default='../raw_data')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_dir', type=str, default='../log')
parser.add_argument('--ckpt_dir', type=str, default='../checkpoint')
parser.add_argument('--result_dir', type=str, default='../result')
parser.add_argument('--csv_dirname', type=str, default='raws_for_MaDE_pseudo')

# GIN args
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--num_mlp_layers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=128)
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
ext_center = args.ext_center

new_subsample = args.new_subsample #  True # 

# making seperate log and checkpoint directory for each setting: e.g. save_model_metrics
log_dir = os.path.join(args.log_dir, args.save_model_metrics)
args.ckpt_dir = os.path.join(args.ckpt_dir, args.save_model_metrics)
result_dir = args.result_dir
raw_data_dir = args.raw_data_dir
data_dir = args.data_dir
csv_dirname = args.csv_dirname

# you can just save constants without dadd_argument
args.n_feats = 5 # it's not the same as n_nodes, like 100 x 100 YAD-HCP case !!
args.ncls = 2 
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

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir,'train'))
    os.makedirs(os.path.join(log_dir,'valid'))

# fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
writer_loss = SummaryWriter(log_dir=os.path.join(log_dir, 'loss'))
writer_metrics = SummaryWriter(log_dir=os.path.join(log_dir, 'metrics'))

#%% Create dataset from csv files, and split into training, validation and test set. 

# set questionnaires, labels and single items
questions = {'PHQ':9,'GAD':7,'STAI':20}#, 'RAS':12, 'RSES':10}
labels = []#'suicidal_idea'] 
single_items = []#['Gender','Age']

# get number of nodes for graph of each subject
num_question = 0
for value in questions.values():
    num_question+=value
n_nodes = num_question+len(single_items)+len(labels)

# set directory
csv_dir = os.path.join(raw_data_dir, csv_dirname) # 'original_refined_survey'

# data preprocess only when new_subsample is activated
if new_subsample:
    df_drop, df = csv2dataframe(questions = questions, labels=labels, single_items = single_items, data_dir = csv_dir)
    
    # make df_test.csv for EBICglasso.R to generate edge feature matrix
    
    '''
    Data structure example:
        pick dataset as:
            df_drop['labelness']['center']
        if unlabelled dataset, (n, n_nodes): questions + single_itmes (= Gender, Age)
        if labelled dataset, (n, n_nodes+2): 2 = MaDE, YAD_group
    '''
    
    unlabeled_centers = ['KAIST', 'SNU']
    for center in unlabeled_centers:
        
        df_drop['Unlabelled'][center].to_csv(os.path.join(csv_dir, f'df_test_{center}.csv'))
        df_test = pd.read_csv(os.path.join(csv_dir, f'df_test_{center}.csv'))
        print(df_test.tail(100))

    make_single_csv(raw_data_dir, df_drop)._total_subj2csv()    
    make_single_csv(raw_data_dir, df_drop)._subsample_balance()
else:
    print("Data directory already exists!!")
    
# partition, label_dicts = split_data(data_dir)
'''
split_ext_data (function): returns subj idx in data_dir, and split them into test (i.e. ext data) and others as train/valid
'''
partition, label_dicts = split_ext_data(data_dir, ext_center)
#%% set dataset and dataloader
dataset = {x: survey_dataset(data_dir, partition[x], label_dicts, n_nodes, seed, use_random_adj=False) for x in ['train', 'valid', 'test']} #  if x == 'test' else True
dataloader = {x: DataLoader(dataset=dataset[x], batch_size = batch_size if x != 'test' else 640, shuffle=True if x =='train' else False, num_workers=ncpu, collate_fn = collate, pin_memory = True) for x in ['train', 'valid', 'test']} 

# get test graph drawn
test_graph = survey_dataset(data_dir, partition['train'], label_dicts, n_nodes, seed, use_random_adj=False).__getitem__(0)[0]
nx.draw_circular(test_graph.to_networkx(), with_labels = True, node_size = n_nodes)

#%% define model with chosen architecture

model = build_model(args)
model = model.to(device)

# for class weighted loss
weights = [1.0, 3.0]
class_weights = torch.FloatTensor(weights).to(device)

# define loss, optimizer, and scheduler
loss_func = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#%% train and valid
epoch_losses = []
epoch_losses_val=[]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total = 0
    correct = 0
    tn, fp, fn, tp = 0, 0, 0, 0
    
    for iter, (bg, label) in enumerate(dataloader['train']):
        
        feats = bg.ndata.pop('n')
        # feats = torch.eye(100)
        # print(feats, feats.size())
        feats, label = feats.to(device), label.to(device)
        
        prediction = model(bg, feats)
        
        # prediction = model(bg)
        loss = loss_func(prediction, label) # + _lambda * weight_norm(model, num_layers, p)
        
        _ , pred_class = torch.max(prediction, dim=1)
        # print("prediction:", pred_class)
        # print("label:", label)
        # print(len(dataloader['train']), label.shape) # len(dataloader) is number of batches=33=32(*32:batch_size)+3=1027
        total += label.size(0)
        correct += torch.sum(pred_class == label).item()
        
        tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
        
        tn+=tn_batch
        fp+=fp_batch
        fn+=fn_batch
        tp+=tp_batch
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader['train']) # same as (iter+1)
    
    acc = correct/total * 100
    print('tn, fp, fn, tp:', tn, fp, fn, tp, 'total:', total)        
    sens = tp/(tp+fn+epsilon) * 100
    spec = tn/(tn+fp+epsilon) * 100
    new_acc = (tp+tn)/(tn+fp+fn+tp+epsilon) * 100
        
    print('Epoch {}/{}, loss {:.4f}, sens {:.4f}, spec {:.4f}, new_acc {:.4f}, acc {:.4f}'.format(epoch+1, num_epochs, epoch_loss, sens, spec, new_acc, acc))
    epoch_losses.append(epoch_loss)
    
    writer_loss.add_scalars('loss/train', {'train loss': np.mean(epoch_losses)}, epoch)
    writer_metrics.add_scalars('metrics/train', {'train sens': sens,
                                              'train spec': spec,
                                              'train acc': new_acc
                                              }, epoch)
    
    # valid for every epoch
    with torch.no_grad():
        model.eval()
        epoch_loss_val=0
        val_total = 0
        val_correct = 0
        tn, fp, fn, tp = 0, 0, 0, 0
        
        for batch, (bg, label) in enumerate(dataloader['valid']):
            
            feats = bg.ndata.pop('n')
            feats, label = feats.to(device), label.to(device)
            
            prediction = model(bg, feats)
            
            val_loss = loss_func(prediction, label) # + _lambda * weight_norm(model, num_layers, p)
            epoch_loss_val += val_loss.detach().item()
            epoch_loss_val /= len(dataloader['valid'])
            
            _ , pred_class = torch.max(prediction, dim=1)
            val_total += label.size(0)
            # print(len(dataloader['valid']), label.shape)
            val_correct += torch.sum(pred_class == label).item()
            
            tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
        
            tn+=tn_batch
            fp+=fp_batch
            fn+=fn_batch
            tp+=tp_batch
        
        val_acc = val_correct/val_total * 100            
        print('tn, fp, fn, tp:', tn, fp, fn, tp, 'total:', val_total)  
        val_sens = tp/(tp+fn+epsilon) * 100
        val_spec = tn/(tn+fp+epsilon) * 100        
        val_new_acc = (tp+tn)/(tn+fp+fn+tp+epsilon) * 100
               
        print(f'valid loss {epoch_loss_val:.4f}, valid sens {val_sens:.4f}, valid spec {val_spec:.4f}, valid new_acc {val_new_acc:.4f}, valid acc {val_acc:.4f}')
        epoch_losses_val.append(epoch_loss_val)
        
        writer_loss.add_scalars('loss/valid', {'valid loss': np.mean(epoch_losses_val)}, epoch)
        writer_metrics.add_scalars('metrics/valid', {'valid sens': val_sens,
                                              'valid spec': val_spec,
                                              'valid acc': val_new_acc
                                              }, epoch)
        
        ### saving best model: loss/sens/spec/acc
        if args.use_best_model:
            val_metrics = {'loss': np.mean(epoch_losses_val), 'sens': val_sens, 'spec': val_spec, 'acc': val_new_acc}
            save_best_model(args, val_metrics, epoch, model, optimizer)
        else:
            if epoch % 10 == 0:
                save(args, model, optimizer, epoch)
        
    scheduler.step()

writer_loss.close()    
writer_metrics.close()
        

#%% test

print(f'############ using {ext_center} as external test set ############')

fn_numpy = lambda x: x.cpu().detach().numpy()

model = build_model(args)
lst_best_models = os.listdir(args.ckpt_dir)

# for model_idx in range(len(lst_best_models)):
    # path_best_model = os.path.join(args.ckpt_dir, lst_best_models[model_idx])
path_best_model = os.path.join(args.ckpt_dir, lst_best_models[-2]) # best model_idx was -2
print(f'loading the last {lst_best_models[-2]} model')
state_dict=torch.load(path_best_model)['net']
model.load_state_dict(state_dict)
model = model.to(device)

lst_y_true = []
lst_y_pred = []

with torch.no_grad():
    model.eval()
    test_loss_arr=[]
    test_correct=0
    test_total=0
    tn, fp, fn, tp = 0, 0, 0, 0
    
    for batch, (bg, label) in enumerate(dataloader['test']):
        
        feats = bg.ndata.pop('n')
        feats, label = feats.to(device), label.to(device)
        prediction = model(bg, feats)
        
        test_loss = loss_func(prediction, label)
        test_loss_arr += [test_loss.item()]
        
        pred_score, pred_class = torch.max(prediction, dim=1)
        
        lst_y_pred.append(fn_numpy(1- torch.sigmoid(prediction[:,0])))
        
        lst_y_true.append(fn_numpy(label))
                
        test_total += label.size(0)
        test_correct += torch.sum(pred_class == label).item()
        
        tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
        
        tn+=tn_batch
        fp+=fp_batch
        fn+=fn_batch
        tp+=tp_batch
    
    # for getting AUC (95% CI)
    y_pred = np.hstack(lst_y_pred)
    y_true = np.hstack(lst_y_true)
    
    auc = roc_auc_score(y_true, y_pred)    
    confidence_lower, confidence_upper = CI_for_AUC(args.result_dir, y_true, y_pred)
    
    test_acc = test_correct/test_total * 100
    print('tn, fp, fn, tp:', tn, fp, fn, tp, 'total:', test_total)  
    test_sens = tp/(tp+fn+epsilon) * 100
    test_spec = tn/(tn+fp+epsilon) * 100        
    test_new_acc = (tp+tn)/(tn+fp+fn+tp+epsilon) * 100
    
    print(f'TEST|Loss:{np.mean(test_loss_arr)}|sens:{test_sens}|spec:{test_spec}|acc:{test_acc}|new_acc:{test_new_acc}|auc:{auc}(95% CI: {confidence_lower} - {confidence_upper}')
    

#%% saliency for single test graph

if args.get_saliency:

    print(f'############ getting a saliency using {ext_center} as external test set ############')
    
    # if args.use_best_model:
    model = build_model(args)
    lst_best_models = os.listdir(args.ckpt_dir)
    path_best_model = os.path.join(args.ckpt_dir, lst_best_models[-1])
    print(f'loading the last {lst_best_models[-1]} model')
    state_dict=torch.load(path_best_model)['net']
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # get a single batch graph
    batch_bg, batch_label = next(dataloader['test'].__iter__())        
    batch_feats = batch_bg.ndata.pop('n')
    batch_feats, batch_label = batch_feats.to(device), batch_label.to(device)
    
    # get a single test graph: test_num indicates the number of graph in the batch, i.e. maximum is 64, or the batch size
    test_num = 1 # 1 ~ 64 
    
    print(batch_feats[(36 * (test_num-1)):(36 * test_num),:].size(), batch_label[test_num-1])
    test_feats = batch_feats[(36 * (test_num-1)):(36 * test_num),:]
    test_feats.requires_grad_()
    test_label = batch_label[test_num-1]
    
    test_bg = dgl.unbatch(bg)[test_num-1]
    test_prediction = model(test_bg, test_feats)
    
    test_pred_score, test_pred_class = torch.max(test_prediction, dim=1) # .unsqueeze(0)
    
    # get a saliency map
    test_pred_score.backward()
    saliency_test = test_feats.grad.data.cpu() # (36,5)
    
    # in case of getting saliency per question as well as answer (0~4 point)
    plt.figure(figsize = (12,4))
    plt.imshow(saliency_test.transpose(1,0), cmap='hot')
    plt.yticks(range(5))
    plt.show()
    
    # in case of getting average saliency per question: getting mean per question
    saliency_test_per_channel = torch.mean(saliency_test, dim=1) # (36)
    # barplot for each questions
    plt.bar(range(len(saliency_test_per_channel)), saliency_test_per_channel)
