# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:03:04 2020

@author: CNDLMembers
"""
import os
import glob
import shutil
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dgl
import networkx as nx
from dgl.model_zoo.chem import GCNClassifier, GATClassifier
import math
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from my_gin_concat import *

def col_normalize(x):
    '''
    x: except 0,1,5,6 (columns with binary labels) columns of original matrix (np.array) with 4 rows
    y: column-wise normalized matrix
    '''
    
    col_questions = list(set(range(x.shape[0])) - set([0,1,5,6]))
    
    x = np.stack([x[col,:] for col in col_questions])
    # print(x.shape) # (19,4)
    
    ls_y = []
    for i in range(x.shape[1]): # 4
        # print(x.shape) # (19,4)
        y = x[:,i] 
        # print(y.shape) # (19,)
        y_max = np.max(np.abs(y))
        # print(y_max)
        y /= y_max
        ls_y.append(y)
    arr_y = np.stack(ls_y)
    # print(arr_y) # (19,4)
    return arr_y


#%%
def row_normalize(x):
    '''
    x: 0,1,5,6 (columns with binary labels) columns of original matrix (np.array) with 4 rows
    y: binarized matrix (np.array) with 2 rows
    '''
    
    ls_y = []
    for row in [0,1,5,6]:
        
        y = x[row,0:2] # choose first and second column
        # print(y) # (2,)
        y /= np.min(np.abs(y))
        y_cp = y.copy()
        y_cp[0] = -y[1]
        y_cp[1] = -y[0]
        ls_y.append(y_cp)
    arr_y = np.stack(ls_y)
    print(arr_y)
    return arr_y


def binarize_saliency(x):
    '''
    x: original matrix (np.array) with 4 rows
    y: binarized matrix (np.array) with 2 rows
    '''
    
    # initialize binarized matrix with the same number of columns as x
    y = np.zeros((x.shape[0], 2))
    
    lst_binary_cols = [0,1,5,6]
    lst_other_cols = [col for col in range(x.shape[0]) if col not in lst_binary_cols]
    
    for col in lst_binary_cols:
        y[col, 0] = x[col, 0]
        y[col, 1] = x[col, 1]
    
    for col in lst_other_cols:
        y[col, 0] = x[col, 0] + x[col, 1]
        y[col, 1] = x[col, 2] + x[col, 3]
        
    return y
        
# zero out saliency matrix for binary columns: gender, site(3 in fact), MaDE, and attempt

def zero_out_saliency(x):
    '''
    x: original matrix (np.array) with 4 rows    
    '''
    
    x[0,2] = 0
    x[0,3] = 0
    
    x[1,2] = 0
    x[1,3] = 0
    
    x[5,2] = 0
    x[5,3] = 0
    
    x[6,2] = 0
    x[6,3] = 0
    
    return x

max_normalize = lambda x: x/np.max(np.abs(x))


def get_saliency_matrix(x):

    '''
    x(list of np.array)
    return args for draw_avg_maps(), draw_binary_maps() and draw_bar_plots()
    '''

    saliency_avg = np.average(x, axis=0) # (23, 4)
    saliency_avg_zero = zero_out_saliency(saliency_avg) # zero out saliency matrix for binary columns
    
    saliency_avg_norm = max_normalize(saliency_avg) # normalize as 2d
    saliency_row_norm = row_normalize(saliency_avg) # normalize to compare sex, MaDE, attempt (+/-) (d/t reversed axis) 
    saliency_col_norm = col_normalize(saliency_avg) # normalize to compare questions
    
    # saliency_row_norm = normalize(saliency_avg, norm='l1', axis=1) # MUST remove norm='max' (l2 by default gives the correct result) !!

    saliency_avg_per_channel = np.average(np.abs(saliency_avg_norm), axis=1) # MUST input saliency_avg_norm instead of saliency_avg (debug 1hr) !!
    # print(saliency_avg_per_channel.shape) # 23
    saliency_avg_per_channel_norm = max_normalize(saliency_avg_per_channel)
    # print(saliency_avg_per_channel_norm.shape)
    # print(saliency_avg_per_channel_norm)
    for i in [0,1,5,6]:
        # for balancing divide by 4 instead of 2 for binary colsd
        saliency_avg_per_channel_norm[i] *= 2
    # print(saliency_avg_per_channel_norm)
        
    return saliency_avg_norm, saliency_row_norm, saliency_col_norm, saliency_avg_per_channel_norm

def draw_avg_maps(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''

    ### draw avg map from true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x.transpose(1,0), cmap='coolwarm') # _norm
    # get node names for xticks from sample csv
    node_names = args.lst_feature
    positions = range(len(node_names))
    plt.title(f'Average saliency for {case_label} cases of {args.label_name}')
    plt.colorbar(label='color')
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(args.n_feats))
    print(f'saving average saliency for {case_label} cases of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average saliency for {case_label} cases of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)
    
def draw_col_maps(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''
    
    # draw binary map for true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x, cmap='coolwarm') # .transpose(1,0)
    # get node names for xticks from sample csv
    node_names = [feat for feat in args.lst_feature if feat not in ['Gender','site','MaDE','suicidal_attempt']]
    positions = range(len(node_names))
    plt.title(f'Average question saliency for {case_label} cases of {args.label_name}')
    plt.colorbar(label='color')
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(args.n_feats))
    print(f'saving question average saliency for {case_label} cases of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average question saliency for {case_label} cases of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)
    
def draw_row_maps(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''
    
    # draw binary map for true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x.transpose(1,0), cmap='OrRd') # coolwarm
    # get node names for xticks from sample csv
    node_names = ['Gender','site','MaDE','suicidal_attempt']
    positions = range(len(node_names))
    plt.title(f'Average binary saliency for {case_label} cases of {args.label_name}')
    plt.colorbar(label='color')
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(2))
    print(f'saving binary average saliency for {case_label} cases of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average binary saliency for {case_label} cases of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)
    
def draw_bar_plots(args, x, avg_figpath, case_label = 'positive'):
    
    '''
    in case of getting average saliency per question: getting mean per question
    '''
    # barplot for each questions
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.title(f'Average saliency for {case_label} cases per question of {args.label_name}')
    plt.bar(range(len(x)), x) # saliency_pos_avg_per_channel_norm
    node_names = args.lst_feature
    positions = range(len(node_names))
    plt.xticks(positions, node_names, rotation=90)
    print(f'saving average saliency for {case_label} cases per question of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average saliency for {case_label} cases per question of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)

#%%

def plot_saliency_maps(args, avg_figpath, saliency_test_dict):
    
    saliency_test_pos = np.stack([t.numpy() for t in saliency_test_dict['positive']])
    saliency_test_neg = np.stack([t.numpy() for t in saliency_test_dict['negative']])
    saliency_test_total = np.concatenate([saliency_test_pos, saliency_test_neg],axis=0)
        
    saliency_avg_norm_pos, saliency_row_norm_pos, saliency_col_norm_pos, saliency_avg_per_channel_norm_pos = get_saliency_matrix(saliency_test_pos)
    saliency_avg_norm_neg, saliency_row_norm_neg, saliency_col_norm_neg, saliency_avg_per_channel_norm_neg = get_saliency_matrix(saliency_test_neg)
    saliency_avg_norm_total, saliency_row_norm_total, saliency_col_norm_total, saliency_avg_per_channel_norm_total = get_saliency_matrix(saliency_test_total)
        
    # saving saliency_test_dict at indiv_figpath
    np.save(os.path.join(avg_figpath, 'saliency_test_dict_positive.npy'), saliency_test_pos) # (173, 23, 4) # not working: np.array(saliency_test_dict['positive']))
    np.save(os.path.join(avg_figpath, 'saliency_test_dict_negative.npy'), saliency_test_neg)
    
    # draw avg maps
    # draw_avg_maps(args, saliency_avg_norm_pos, avg_figpath, case_label = 'positive')
    # draw_avg_maps(args, saliency_avg_norm_neg, avg_figpath, case_label = 'negative')
    draw_avg_maps(args, saliency_avg_norm_total, avg_figpath, case_label = 'total_avg')
    # draw_avg_maps(args, saliency_row_norm_total, avg_figpath, case_label = 'total_row')
    # draw_avg_maps(args, saliency_col_norm_total, avg_figpath, case_label = 'total_col')
    
    # draw binary maps
    # draw_binary_maps(args, saliency_avg_bin_norm_pos, avg_figpath, case_label = 'positive')
    # draw_binary_maps(args, saliency_avg_bin_norm_neg, avg_figpath, case_label = 'negative')
    draw_row_maps(args, saliency_row_norm_total, avg_figpath, case_label = 'total')
    draw_col_maps(args, saliency_col_norm_total, avg_figpath, case_label = 'total')
    
    # draw barplot for each questions in case of getting average saliency per question: getting mean per question
    # draw_bar_plots(args, saliency_avg_per_channel_norm_pos, avg_figpath, case_label = 'positive')
    # draw_bar_plots(args, saliency_avg_per_channel_norm_neg, avg_figpath, case_label = 'negative')
    draw_bar_plots(args, saliency_avg_per_channel_norm_total, avg_figpath, case_label = 'total')
    
    ### saving all matrix for plots as txt
    save_arrs = [saliency_avg_norm_total , saliency_row_norm_total, saliency_col_norm_total, saliency_avg_per_channel_norm_total]
    save_names = ['saliency_avg_norm_total' , 'saliency_row_norm_total', 'saliency_col_norm_total', 'saliency_avg_per_channel_norm_total']
    for name, arr in zip(save_names, save_arrs):
        np.savetxt(os.path.join(avg_figpath, f'{name}.txt'), arr, delimiter = ',')
    
#%% for test_idea2wks_ensemble_single_case_saliency_test.py

def draw_col_maps_single(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''
    
    # draw binary map for true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x, cmap='coolwarm') # .transpose(1,0)
    # get node names for xticks from sample csv
    node_names = [feat for feat in args.lst_feature if feat not in ['Gender','site','MaDE','suicidal_attempt']]
    positions = range(len(node_names))
    plt.title(f'Average question saliency for {case_label} case of {args.label_name}')
    plt.colorbar(label='color')
    
    # display true and pred_label with prediction score
    plt.gcf().text(0.9, 0.6, f'true label:\n{args.true_pred_score[0]}')
    plt.gcf().text(0.9, 0.4, f'pred label:\n{args.true_pred_score[1]}')
    plt.gcf().text(0.9, 0.2, f'pred score (%):\n{args.true_pred_score[2]:.2f}')
    
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(args.n_feats))
    print(f'saving question average saliency for {case_label} case of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average question saliency for {case_label} case of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)
    
def draw_row_maps_single(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''
    
    # draw binary map for true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x.transpose(1,0), cmap='OrRd') # coolwarm
    # get node names for xticks from sample csv
    node_names = ['Gender','site','MaDE','suicidal_attempt']
    positions = range(len(node_names))
    plt.title(f'Average binary saliency for {case_label} case of {args.label_name}')
    plt.colorbar(label='color')
    
    # display true and pred_label with prediction score
    plt.gcf().text(0.9, 0.6, f'true label:\n{args.true_pred_score[0]}')
    plt.gcf().text(0.9, 0.4, f'pred label:\n{args.true_pred_score[1]}')
    plt.gcf().text(0.9, 0.2, f'pred score (%):\n{args.true_pred_score[2]:.2f}')
    
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(2))
    print(f'saving binary average saliency for {case_label} case of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average binary saliency for {case_label} case of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)
    
def draw_bar_plots_single(args, x, avg_figpath, case_label = 'positive'):
    
    '''
    in case of getting average saliency per question: getting mean per question
    '''
    # barplot for each questions
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.title(f'Average saliency for {case_label} case per question of {args.label_name}')
    plt.bar(range(len(x)), x) # saliency_pos_avg_per_channel_norm
    
    # display true and pred_label with prediction score
    plt.gcf().text(0.9, 0.6, f'true label:\n{args.true_pred_score[0]}')
    plt.gcf().text(0.9, 0.4, f'pred label:\n{args.true_pred_score[1]}')
    plt.gcf().text(0.9, 0.2, f'pred score (%):\n{args.true_pred_score[2]:.2f}')
    
    node_names = args.lst_feature
    positions = range(len(node_names))
    plt.xticks(positions, node_names, rotation=90)
    print(f'saving average saliency for {case_label} case per question of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average saliency for {case_label} case per question of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)

    
def draw_avg_maps_single(args, x, avg_figpath, case_label = 'positive'):
    '''
    x(np.array): matrix to draw
    '''

    ### draw avg map from true positive cases
    plt.ioff()
    fig = plt.figure(figsize = (12,4))
    plt.imshow(x.transpose(1,0), cmap='coolwarm') # _norm
    # get node names for xticks from sample csv
    node_names = args.lst_feature
    positions = range(len(node_names))
    plt.title(f'Average saliency for {case_label} case of {args.label_name}')
    plt.colorbar(label='color')
    
    # display true and pred_label with prediction score
    plt.gcf().text(0.9, 0.6, f'true label:\n{args.true_pred_score[0]}')
    plt.gcf().text(0.9, 0.4, f'pred label:\n{args.true_pred_score[1]}')
    plt.gcf().text(0.9, 0.2, f'pred score (%):\n{args.true_pred_score[2]:.2f}')
    
    plt.xticks(positions, node_names, rotation=90)
    plt.yticks(range(args.n_feats))
    print(f'saving average saliency for {case_label} case of {args.label_name}.png')
    plt.savefig(os.path.join(avg_figpath, f'Average saliency for {case_label} case of {args.label_name}.png'), dpi=300)
    # plt.show()
    plt.close(fig)

def plot_saliency_maps_for_single(args, avg_figpath, saliency_test_dict):
    
    '''
    plot_saliency_maps function for single case of test set
    '''
    
    if saliency_test_dict['positive'] != []:    
        saliency_test_total = np.stack([t.numpy() for t in saliency_test_dict['positive']])
        
    if saliency_test_dict['negative'] != []:
        saliency_test_total = np.stack([t.numpy() for t in saliency_test_dict['negative']])
        
    saliency_avg_norm_total, saliency_row_norm_total, saliency_col_norm_total, saliency_avg_per_channel_norm_total = get_saliency_matrix(saliency_test_total)
        
    # saving saliency_test_dict at indiv_figpath
    np.save(os.path.join(avg_figpath, 'saliency_test_dict.npy'), saliency_test_total) # (173, 23, 4) # not working: np.array(saliency_test_dict['positive']))
    
    # draw avg maps
    draw_avg_maps_single(args, saliency_avg_norm_total, avg_figpath, case_label = 'single')
    
    # draw binary maps
    draw_row_maps_single(args, saliency_row_norm_total, avg_figpath, case_label = 'single')
    draw_col_maps_single(args, saliency_col_norm_total, avg_figpath, case_label = 'single')
    
    # draw barplot for each questions in case of getting average saliency per question: getting mean per question
    draw_bar_plots_single(args, saliency_avg_per_channel_norm_total, avg_figpath, case_label = 'single')
    
    ### saving all matrix for plots as txt
    save_arrs = [saliency_avg_norm_total , saliency_row_norm_total, saliency_col_norm_total, saliency_avg_per_channel_norm_total]
    save_names = ['saliency_avg_norm_total' , 'saliency_row_norm_total', 'saliency_col_norm_total', 'saliency_avg_per_channel_norm_total']
    for name, arr in zip(save_names, save_arrs):
        np.savetxt(os.path.join(avg_figpath, f'{name}.txt'), arr, delimiter = ',')
    
    
#%%
def under_sample(X,y, balance_ratio):
    print(f'imbalance ratio is {len(y) / sum(y)}')
    lst_pos_idx = np.array([y_idx for y_idx, y_elem in enumerate(y) if y_elem == 1], dtype=int)
    lst_neg_idx = np.array([y_idx for y_idx, y_elem in enumerate(y) if y_elem == 0], dtype=int)
    # lst_total = np.arange(0,X.shape[0])    
    lst_neg_idx_undersampled = np.random.choice(lst_neg_idx, size=int(sum(y) * balance_ratio), replace=False)
    undersampled_idx = np.concatenate((lst_pos_idx, lst_neg_idx_undersampled), axis=0) # just concat makes all pos idx in the front, and all neg_idx in the last  
    X_undersampled = X[undersampled_idx]
    y_undersampled = y[undersampled_idx]
    
    print(f'number of undersampled samples: pos = {sum(y_undersampled)}, neg = {len(y_undersampled) - sum(y_undersampled)}')
    return X_undersampled, y_undersampled 

def get_feature_columns(questions, single_items, label_lst):
    '''
    make input feature columns (i.e. questions, single_items, label_lst) into a feature list
    '''
    
    lst_questions = []
    for key in questions.keys():
        for i in range(questions[key]):
            if key in ['STAI','RAS','RSES']:
                lst_questions.append(f'{key}_{i+1:02d}') # STAI_01, not STAI_1
            else:
                lst_questions.append(f'{key}_{i+1}')

    lst_feature_label = single_items + lst_questions + label_lst # get features and labels as well
    
    return lst_feature_label

# make dataset as train, valid, and test directories with split_ratio as single csv files

class make_datasets(object):
    def __init__(self, args, questions, single_items, label_lst, lst_filename):
        self.questions = questions
        self.single_items = single_items
        self.label_lst = label_lst
        
        # making feature list using above information
        self.lst_feature_label = get_feature_columns(questions, single_items, label_lst)
        self.df_total_drop = self._make_dataset(args, lst_filename)       
        
    def _make_dataset(self, args, lst_filename):
        
        # make train and valid set using unlabeled csv files
        lst_df_total_feature = []
        
        for total_filename in lst_filename:
            
            print(f'making total data set with {total_filename}')
            df_total = pd.read_csv(os.path.join(args.csv_dir, total_filename), encoding='CP949')
            df_total_feature = df_total[self.lst_feature_label]
            df_total_feature = df_total_feature.replace('empty', None) 
            lst_df_total_feature.append(df_total_feature)
        
        df_total = pd.concat(lst_df_total_feature, ignore_index=True)            
                
        # missing values in features_with_total columns are coded as 'empty' (string), not NaN: replace 'empty' as 'NaN'
        df_total_drop = df_total.dropna(axis=0) # MAKE SURE to reassign to df_total_feature !! (i.e. not just df_total_feature.replace('empty', None), but df_total_feature = df_total_feature.replace('empty', None) ) 
        df_total_drop.reset_index(drop=True, inplace=True) # reset index after dropna: MUST specify options !!
        
        args.labels = np.array(df_total_drop[args.label_name], dtype=int)
        print(f'labels from make_dataset: {sum(args.labels)}/{len(args.labels)}')
        
        print(f'number of input features: {args.n_nodes}')
        print(f'dropped NA in total set: {df_total.shape[0]} to {df_total_drop.shape[0]}')
        
        return df_total_drop
    
# convert total scores into one-hot vector using quantiles
def convert_total_quantile(args, df_total_drop, dataset=None):
    
    if dataset:
        
        # saving quantile values of dataset as dict
        quantile_values={} 
        save_quan_path = os.path.join(args.csv_dir, f'quantile_values_{dataset}.pkl')
        
        features_with_reversed_total = ['RAS_total', 'RSES_total']
        for feature in features_with_reversed_total:
        
            # convert from str into int
            df_total_drop[feature] = df_total_drop[feature].astype(int)
            
            # get quantile values from total dataset
            q1 = df_total_drop[feature].quantile(0.25)
            q2 = df_total_drop[feature].quantile(0.50)
            q3 = df_total_drop[feature].quantile(0.75)
            
            # print(f"{feature}: 0.25, 0.50, 0.75 quantile values are {q1},{q2},{q3}")
                
            df_total_drop[feature][df_total_drop[feature] < q1] = 4 # 1 # reverse the order
            df_total_drop[feature][(df_total_drop[feature] < q2) & (df_total_drop[feature] >= q1)] = 3 # 2
            df_total_drop[feature][(df_total_drop[feature] < q3) & (df_total_drop[feature] >= q2)] = 2 # 3
            df_total_drop[feature][df_total_drop[feature] >= q3] = 1 # 4
            
            quantile_values[feature] = [q1, q2, q3]
            
        for feature in ['STAI_X1_total']:
            
            # convert from str into int
            df_total_drop[feature] = df_total_drop[feature].astype(int)
            
            # get quantile values from total dataset
            q1 = df_total_drop[feature].quantile(0.25)
            q2 = df_total_drop[feature].quantile(0.50)
            q3 = df_total_drop[feature].quantile(0.75)
        
            # print(f"{feature}: 0.25, 0.50, 0.75 quantile values are {q1},{q2},{q3}")
                
            df_total_drop[feature][df_total_drop[feature] < q1] = 1
            df_total_drop[feature][(df_total_drop[feature] < q2) & (df_total_drop[feature] >= q1)] = 2
            df_total_drop[feature][(df_total_drop[feature] < q3) & (df_total_drop[feature] >= q2)] = 3
            df_total_drop[feature][df_total_drop[feature] >= q3] = 4
            
            quantile_values[feature] = [q1, q2, q3]
        
        # saving quantile values of dataset as dict        
        with open(save_quan_path,'wb') as f:
            pickle.dump(quantile_values, f)
            
    else:
        
        # loading quantile values of dataset as dict
        save_quan_path = os.path.join(args.csv_dir, 'quantile_values_train.pkl') # quantile_values_valid.pkl
    
        with open(save_quan_path,'rb') as f:
            quantile_values = pickle.load(f)        
            
        features_with_reversed_total = ['RAS_total', 'RSES_total']
        for feature in features_with_reversed_total:
        
            # convert from str into int
            df_total_drop[feature] = df_total_drop[feature].astype(int)
            
            # get quantile values from total dataset
            print(f'{feature}: using quantile_values obtained from train dataset')
            q1 = quantile_values[feature][0] # df_total_drop[feature].quantile(0.25)
            q2 = quantile_values[feature][1] # df_total_drop[feature].quantile(0.50)
            q3 = quantile_values[feature][2] # df_total_drop[feature].quantile(0.75)
            
            # print(f"{feature}: 0.25, 0.50, 0.75 quantile values are {q1},{q2},{q3}")
                
            df_total_drop[feature][df_total_drop[feature] < q1] = 4 # 1 # reverse the order
            df_total_drop[feature][(df_total_drop[feature] < q2) & (df_total_drop[feature] >= q1)] = 3 # 2
            df_total_drop[feature][(df_total_drop[feature] < q3) & (df_total_drop[feature] >= q2)] = 2 # 3
            df_total_drop[feature][df_total_drop[feature] >= q3] = 1 # 4
            
            quantile_values[feature] = [q1, q2, q3]
            
        for feature in ['STAI_X1_total']:
            
            # convert from str into int
            df_total_drop[feature] = df_total_drop[feature].astype(int)
            
            # get quantile values from total dataset
            print(f'{feature}: using quantile_values obtained from train dataset')
            q1 = quantile_values[feature][0] # df_total_drop[feature].quantile(0.25)
            q2 = quantile_values[feature][1] # df_total_drop[feature].quantile(0.50)
            q3 = quantile_values[feature][2] # df_total_drop[feature].quantile(0.75)
        
            # print(f"{feature}: 0.25, 0.50, 0.75 quantile values are {q1},{q2},{q3}")
                
            df_total_drop[feature][df_total_drop[feature] < q1] = 1
            df_total_drop[feature][(df_total_drop[feature] < q2) & (df_total_drop[feature] >= q1)] = 2
            df_total_drop[feature][(df_total_drop[feature] < q3) & (df_total_drop[feature] >= q2)] = 3
            df_total_drop[feature][df_total_drop[feature] >= q3] = 4
        
    return df_total_drop
        
def split_into_single_csv(df, data_dir):
    print("saving df into a single csv files in data directory")
    for subj_idx in range(len(df)):
        df.iloc[subj_idx].to_csv(os.path.join(data_dir, f'subj_{subj_idx+1}.csv'), index=True)
         
#%% class Dataset

class survey_dataset(Dataset):
    def __init__(self, args, data_dir):
        self.data_dir = data_dir # not args.data_dir, but train_data_dir, valid_data_dir
        self.csv_dir = args.csv_dir # not args.data_dir, but train_data_dir, valid_data_dir
        self.n_nodes = args.n_nodes
        self.n_feats = args.n_feats # size of the scale: i.e. 0~5 --> num_scale = 6, which makes an error (debug for 3 hours): because RAS and RSES scale is not 0~4 but 0~5 !!
        self.label_name = args.label_name
        self.lst_feature = args.lst_feature
        self.thr = args.thr
        
    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def __getitem__(self, index):
        lst = os.listdir(self.data_dir)
        subj_name = lst[index] # NEVER change this idx to index !!! (debug for 3 hours)
        
        # preprocessing node features obtained from csv file of each subject
        pd_node_ft = pd.read_csv(os.path.join(self.data_dir, subj_name), index_col=0) # set index column with first column
        # print(pd_node_ft)
        # print(pd_node_ft.loc[self.lst_feature])
        node_ft = np.array(pd_node_ft.loc[self.lst_feature]) # (59, 1) # removing labels in the last, not to be in node features
        # print(node_ft)
        
        targets = np.array(node_ft.astype(int)).reshape(-1) # reshape(-1) makes dim = 0: i.e. (n_nodes, )
        node_ft_one_hot = np.eye(self.n_feats)[targets-1] # targets should start from 0, not 1 as previously coded
        node_ft_one_hot = torch.from_numpy(node_ft_one_hot) # (n_nodes, n_feats)
        node_ft_one_hot = node_ft_one_hot.type(torch.FloatTensor)

        # get label from filename of each subject
        label = np.array(pd_node_ft.loc[self.label_name], dtype = np.uint8)
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor) # LongTensor  
        
        # get group-wise adj mtx obtained from EBICglasso of R
        df_from = pd.read_csv(os.path.join(self.csv_dir, f'df_from_{self.label_name}_thr{self.thr}.csv')) 
        df_to = pd.read_csv(os.path.join(self.csv_dir, f'df_to_{self.label_name}_thr{self.thr}.csv'))
        df_weight = pd.read_csv(os.path.join(self.csv_dir, f'df_weight_{self.label_name}_thr{self.thr}.csv'))
            
        # convert dataframe into array
        adj_from_np = np.array(df_from.iloc[:,1] -1) # because the node number of DGLGraph starts with 0, not 1
        adj_to_np = np.array(df_to.iloc[:,1] - 1) 
        adj_weight_np = np.array(df_weight.iloc[:,1])
        adj_from = torch.from_numpy(adj_from_np)
        adj_to = torch.from_numpy(adj_to_np)
        adj_weight = torch.from_numpy(adj_weight_np).unsqueeze(-1)
        
        # make A, and X into a graph g
        g = dgl.DGLGraph()
        g.add_nodes(self.n_nodes, {'n': node_ft_one_hot})
        g.add_edges(adj_from, adj_to, {'e':adj_weight})
        
        self.g = g
        self.label = label
        
        return self.g, self.label

      
#%%
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# build model with chosen architecture
def build_model(args):
    if args.model_arch == 0: 
        model = GCNClassifier(in_feats=args.n_feats,
                        gcn_hidden_feats=[32,32],
                        n_tasks=args.ncls,
                        classifier_hidden_feats=16,
                        dropout=0.5)
    elif args.model_arch == 1: 
        model = GATClassifier(in_feats=args.n_feats,
                    gat_hidden_feats=[32,32],
                    num_heads=[4,4], 
                    n_tasks=args.ncls,
                    classifier_hidden_feats=8
                    )
    elif args.model_arch == 2: 
        model = GIN(
            args.num_layers, args.num_mlp_layers,
            args.n_feats, args.hidden_dim, args.ncls,
            args.final_dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type)
    else:
        print("not implemented model")
    return model

def save(args, net, optim, epoch):
    
    if args.use_best_model:        
        print(f'saving best model at epoch {epoch+1} with {args.save_model_metrics} of {args.best_metrics}')    
    else:
        print(f'saving the model at epoch {epoch+1} for every 10 epoch')
        
    torch.save({'net':net.state_dict(), 'optim':optim.state_dict()},
               f'./{args.ckpt_dir}/model_epoch{epoch+1}_{args.save_model_metrics}_{args.best_metrics}.pth')
    
fn_tonumpy=lambda x: x.cpu().detach().numpy()

def evaluate(model, batched_graph, features, label):
    model.eval()
    sum_label=0
    with torch.no_grad():
        prediction = model(batched_graph, features)
        # _, pred_class = torch.max(prediction, dim=1)
        pred_class = (torch.sigmoid(prediction)>0.5).float()
        pred_class = fn_tonumpy(pred_class).astype(int)
        label = fn_tonumpy(label).astype(int)
        print('label:',label, 'num_positive:', sum(label), 'num_total:', len(label))
        print('pred_class:', pred_class)
        # print('prediction:', prediction)#, 'size:', prediction.size()) # (32,2)
        sum_label = label+pred_class
        tn=len(sum_label[np.where(sum_label==0)])
        tp=len(sum_label[np.where(sum_label==2)])
        tmp=len(sum_label[np.where(sum_label==1)])
        fn=len(set(*np.where(sum_label==1))-set(*np.where(label==0)))
        fp=tmp-fn
        print("batch tn, fp, fn, tp:", tn, fp, fn, tp)
    return tn, fp, fn, tp

#%% bootstrap for AUC CI in ROC

def CI_for_AUC(auc_path, y_true, y_pred):
    '''
    y_true, y_pred (list)
    '''
    n_bootstraps = 1000
    bootstrapped_scores = []
    seed = 0
    rng = np.random.RandomState(seed)
    
    # first convert list into a np.array
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print(f'Bootstrap {i+1} ROC area {score:0.3f}')
                             
    # get CI from AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    # print(f'Confidence interval for the score: [{confidence_lower:0.3f} - {confidence_upper:0.3f}]')
    
    # plot histogram for bootstrapped scores of AUCs
    plt.ioff()
    plt.hist(bootstrapped_scores, bins=50)
    plt.title('Histogram of the bootstrapped AUCs')
    plt.savefig(os.path.join(auc_path, 'Histogram of the bootstrapped AUCs.png'),dpi=300)
    # plt.show()
    plt.close()
    
    return confidence_lower, confidence_upper

#%% get t-SNE from the last hidden layer output

def draw_tsne_plot(args, x, y, label_name):
    tsne = TSNE(n_components = 2, perplexity=30, learning_rate = 100, metric = 'cosine') # 50
    
    # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
    # x = np.array(x).reshape(-1, 1) # only when tsne with output itself rather than last hidden output
    transformed = tsne.fit_transform(x)
    
    # print('predictions:', x)
    # print('transformed:', transformed)
    xs = transformed[:,0] # because TSNE defaultly reduced the dimension of the given data into 2
    ys = transformed[:,1]
    # print('xs, ys:', xs, ys)
    
    # label = y.cpu().numpy()
    label = np.array(y)
    
    plt.scatter(xs, ys, c = label)
    
    print(f'number of tsne points: {len(xs)}')
    
    # for i in range(len(xs)):        
    #     plt.annotate(i+1, (xs[i], ys[i])) # point annotation starts from 1, not 0
