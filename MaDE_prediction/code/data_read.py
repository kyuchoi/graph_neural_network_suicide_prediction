#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:45:57 2020

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

# directory
parser.add_argument('--new_subsample', action="store_true", default=False, help='subsample new balanced datasets to train')
parser.add_argument('--raw_data_dir', type=str, default='../raw_data')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_dir', type=str, default='../log')
parser.add_argument('--ckpt_dir', type=str, default='../checkpoint')
parser.add_argument('--result_dir', type=str, default='../result')
parser.add_argument('--csv_dirname', type=str, default='raws_for_MaDE_pseudo')

args = parser.parse_args()

batch_size = args.batch_size
seed = args.seed
lr = args.lr
ncpu = args.ncpu
num_epochs = args.num_epochs
args.best_epoch = 0
ext_center = args.ext_center

new_subsample = True # args.new_subsample # 

# making seperate log and checkpoint directory for each setting: e.g. save_model_metrics
log_dir = os.path.join(args.log_dir, args.save_model_metrics)
args.ckpt_dir = os.path.join(args.ckpt_dir, args.save_model_metrics)
result_dir = args.result_dir
raw_data_dir = args.raw_data_dir
data_dir = args.data_dir
csv_dirname = args.csv_dirname

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