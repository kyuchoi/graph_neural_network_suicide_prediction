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
args.new_subsample = True # False #   
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

raw_data_dir = args.raw_data_dir
args.data_dir = os.path.join(args.data_dir, f'{args.sample_opt}')

# define train, valid, and test directory
train_data_dir = os.path.join(args.data_dir, 'train')
valid_data_dir = os.path.join(args.data_dir, 'valid')
test_data_dir = os.path.join(args.data_dir, 'test')
csv_dir = os.path.join(raw_data_dir, 'data_for_smote')

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
    
    
