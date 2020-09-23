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
from gin import *
    
parser = argparse.ArgumentParser()
# training
parser.add_argument('--model_arch', type=int, default=2, help='0 for GCN, 1 for GAT, and 2 for GIN')
parser.add_argument('--batch-size', type=int, default=640)
parser.add_argument('--seed', type=int, default=9877)
parser.add_argument('--ncpu', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_best_model', action="store_true", default=False, help='use save_best_model function with various metrics')
parser.add_argument('--save_model_metrics', type=str, default='loss', help='choose among loss/sens/spec/acc')
parser.add_argument('--ext_center', type=str, default='Gachon') # SMC # Gachon # KAIST
parser.add_argument('--single_prediction', action="store_true", default=False, help='get MaDE pseudolabel for a single test case')
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


args.single_prediction = True

batch_size = args.batch_size
seed = args.seed
lr = args.lr
ncpu = args.ncpu
num_epochs = args.num_epochs
args.best_epoch = 0
ext_center = args.ext_center

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
csv_dir = os.path.join(raw_data_dir, csv_dirname) # original_refined_survey
args.csv_dir = csv_dir

#%%
single_item = ['No', 'center']

# load the best model
model = build_model(args)
lst_best_models = os.listdir(args.ckpt_dir)
path_best_model = os.path.join(args.ckpt_dir, lst_best_models[-1])
print(f'loading the last {lst_best_models[-1]} model')
state_dict=torch.load(path_best_model)['net']
model.load_state_dict(state_dict)
model = model.to(device)

#%% set test directory

if not args.single_prediction:
    
    lst_test_filename = ['Unlabelled_KAIST_original.csv', 'Unlabelled_SNU_original.csv'] # MAKE sure to the loaded filename: with or without _original?
    
    test_data_dir = os.path.join(csv_dir, 'test_pseudo')
    args.test_data_dir = test_data_dir 
    test_data_dir = args.test_data_dir 
    
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)    
    
    # make csv files in test_pseudo directory under csv_dir
    make_dataset_pseudo(questions, single_item).make_test(args, lst_test_filename) # NEVER uncomment this !!
    
    test_set_pseudo = survey_dataset_pseudo(args, test_data_dir)#.__getitem__(0)
    dataloader_pseudo = DataLoader(dataset=test_set_pseudo, batch_size = batch_size, shuffle=False, num_workers=ncpu, collate_fn = collate, pin_memory = True)
    
    # get predicted MaDE labels for unlabeled_KAIST, and SNU csv files
    lst_pred_class = []
    lst_label = []
    for batch, (bg, label) in enumerate(dataloader_pseudo):
            
        feats = bg.ndata.pop('n')
        feats = feats.to(device)
        prediction = model(bg, feats)
        pred_score, pred_class = torch.max(prediction, dim=1)
        # print(pred_class) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            # 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
        np_pred_class = pred_class.cpu().detach().numpy()
        np_label = label.cpu().detach().numpy()
        print(f'batch_num:{batch}/{len(dataloader_pseudo)}, label:{list(np_label)}, pred: {list(np_pred_class)}')
        '''
        Here, np_label means ID (No) of subj (NOT!! MaDE pseudolabels), and pred_class means MaDE pseudolabels
        '''
        lst_pred_class.extend(list(np_pred_class))
        lst_label.extend(list(np_label))
    print(len(lst_pred_class))
    print(len(lst_label))
    
    # save
    original_dir = csv_dir
    np.save(os.path.join(original_dir, 'lst_pred_class.npy'), np.array(lst_pred_class))
    np.save(os.path.join(original_dir, 'lst_label.npy'), np.array(lst_label))
    
    # make list for each center
    
    idx_label_KAIST = [idx for idx, label in enumerate(lst_label) if label < 100000] # label means ID (No) of subj
    idx_label_SNU = [idx for idx, label in enumerate(lst_label) if label > 100000]
    
    dict_lst_label = {}
    dict_lst_label['KAIST'] = [lst_label[idx] for idx in idx_label_KAIST]
    dict_lst_label['SNU'] = [lst_label[idx] for idx in idx_label_SNU]
    dict_lst_pred_class = {}
    dict_lst_pred_class['KAIST'] = [lst_pred_class[idx] for idx in idx_label_KAIST]
    dict_lst_pred_class['SNU'] = [lst_pred_class[idx] for idx in idx_label_SNU]
      
    # get new predicted MaDE label for a single large idea2wks csv file
    
    idea_dst = csv_dir # r'D:\Data\YAD_survey\raw_data\idea2wks_refined_survey' # 
    
    for test_file in lst_test_filename:
        idea2wks_path = os.path.join(idea_dst, test_file) 
        pd_idea2wks = pd.read_csv(idea2wks_path, encoding='CP949')#, index_col=0)
        center = test_file.split('_')[1].split('.')[0] # KAIST
        
        for idx, key_no in enumerate(dict_lst_label[center]):
            MaDE_label = dict_lst_pred_class[center][idx]
            print(f'{idx}/{len(dict_lst_label[center])}th label of {center} of No. {key_no}: {MaDE_label}')
            pd_idea2wks.loc[pd_idea2wks['No'] == key_no, 'MaDE'] = MaDE_label # No
        MaDE_pseudo_path = os.path.join(idea_dst, f'Unlabelled_{center}_MaDE_pseudo.csv')
        pd_idea2wks.to_csv(MaDE_pseudo_path, index=False)

else:
    
    # predicting a single test case
    lst_test_filename = ['single_case_original.csv'] # MAKE sure to the loaded filename: with or without _original?
    
    test_data_dir = os.path.join(csv_dir, 'single_test_pseudo')
    args.test_data_dir = test_data_dir 
    test_data_dir = args.test_data_dir 
    
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)    
    
    # make csv files in test_pseudo directory under csv_dir
    make_dataset_pseudo(questions, single_item).make_test(args, lst_test_filename) # NEVER uncomment this !!
    
    test_set_pseudo = survey_dataset_pseudo(args, test_data_dir)#.__getitem__(0)
    dataloader_pseudo = DataLoader(dataset=test_set_pseudo, batch_size = 1, shuffle=False, num_workers=ncpu, collate_fn = collate, pin_memory = True)
    
    # get predicted MaDE labels for unlabeled_KAIST, and SNU csv files
    lst_pred_class = []
    lst_label = []
    for batch, (bg, label) in enumerate(dataloader_pseudo):
            
        feats = bg.ndata.pop('n')
        feats = feats.to(device)
        prediction = model(bg, feats)
        pred_score, pred_class = torch.max(prediction, dim=1)
        # print(pred_class) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            # 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
        np_pred_class = pred_class.cpu().detach().numpy()
        np_label = label.cpu().detach().numpy()
        print(f'batch_num:{batch+1}/{len(dataloader_pseudo)}, No.:{list(np_label)}, pred: {list(np_pred_class)}')
        '''
        Here, np_label means ID (No) of subj (NOT!! MaDE pseudolabels), and pred_class means MaDE pseudolabels
        '''
        lst_pred_class.extend(list(np_pred_class))
        lst_label.extend(list(np_label))
    # print(len(lst_pred_class))
    # print(len(lst_label))
    
    # make list for TEST center
    
    idx_label_TEST = [idx for idx, label in enumerate(lst_label)]
    
    dict_lst_label = {}
    dict_lst_label['TEST'] = [lst_label[idx] for idx in idx_label_TEST]
    dict_lst_pred_class = {}
    dict_lst_pred_class['TEST'] = [lst_pred_class[idx] for idx in idx_label_TEST]
      
    # get new predicted MaDE label for a single large idea2wks csv file
    
    idea_dst = csv_dir # r'D:\Data\YAD_survey\raw_data\idea2wks_refined_survey' # 
    
    for test_file in lst_test_filename:
        idea2wks_path = os.path.join(idea_dst, test_file) 
        pd_idea2wks = pd.read_csv(idea2wks_path, encoding='CP949')#, index_col=0)
        center = 'TEST' # test_file.split('_')[1].split('.')[0] # KAIST
        
        for idx, key_no in enumerate(dict_lst_label[center]):
            MaDE_label = dict_lst_pred_class[center][idx]
            print(f'{idx+1}/{len(dict_lst_label[center])}th label of {center} of No. {key_no}: {MaDE_label}')
            pd_idea2wks.loc[pd_idea2wks['No'] == key_no, 'MaDE'] = MaDE_label # No
        MaDE_pseudo_path = os.path.join(result_dir, f'single_case_MaDE_pseudo.csv')
        pd_idea2wks.to_csv(MaDE_pseudo_path, index=False)
        
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
    
    test_pred_score, test_pred_class = torch.max(test_prediction.unsqueeze(0), dim=1)
    
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
