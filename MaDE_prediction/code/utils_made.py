# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:03:04 2020

@author: CNDLMembers
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dgl
import networkx as nx
from dgl.model_zoo.chem import GCNClassifier, GATClassifier
from sklearn.metrics import *
from sklearn.manifold import TSNE
from gin import *
import matplotlib.pyplot as plt

#%%

class make_dataset_pseudo(object):
    def __init__(self, questions, single_item):
        self.questions = questions
        
        # making feature list using above information
        self.question_lst=[]
        for key in questions.keys():
            for i in range(questions[key]):
                if key in ['STAI','RAS','RSES']:
                    self.question_lst.append(f'{key}_{i+1:02d}') # STAI_01, not STAI_1
                else:
                    self.question_lst.append(f'{key}_{i+1}')
        feature_lst = self.question_lst + single_item
        self.feature_lst = feature_lst
                        
    def make_test(self, args, lst_test_filename):
   
        # make all labeled csv files into a test set
        
        lst_df_test_feature=[]
        for test_filename in lst_test_filename:
            print(f'making test set with {test_filename}')
            center = os.path.basename(test_filename).split('_')[1].split('.')[0] # KAIST
            df_test = pd.read_csv(os.path.join(args.csv_dir, test_filename), encoding='CP949')
            df_test['center'] = center
            # print(df_test.columns)
            # print(df_test['center'])
            # print(self.feature_lst)
            lst_df_test_feature.append(df_test[self.feature_lst])
            
        df_test_total = pd.concat(lst_df_test_feature, ignore_index=True)
        df_test_drop = df_test_total.dropna(axis=0)
        df_test_drop.reset_index(drop=True, inplace=True) # reset index after dropna: MUST specify options !!
        df_test_drop.to_csv(os.path.join(args.csv_dir, 'total_test_drop.csv'), index=False) # False index when saving df as total csv
        
        print(f'dropped NA in test set: {df_test_total.shape[0]} to {df_test_drop.shape[0]}')
        
        # make df into a single csv files in test directory    
        for subj_idx in range(len(df_test_drop)):
            chosen_center = df_test_drop.iloc[subj_idx]['center']
            # print(chosen_center) # KAIST
            df_test_drop.iloc[subj_idx].to_csv(os.path.join(args.test_data_dir, f'{chosen_center}_subj_{subj_idx+1}.csv'), index=True) # True index when saving df as a single csv
                        
class survey_dataset_pseudo(Dataset):
    def __init__(self, args, data_dir):
        self.data_dir = data_dir # not args.data_dir, but train_data_dir, valid_data_dir
        self.raw_data_dir = args.raw_data_dir # not args.data_dir, but train_data_dir, valid_data_dir
        self.n_nodes = 36
        self.n_feats = 5
        
    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def __getitem__(self, index):
        lst = os.listdir(self.data_dir)
        subj_name = lst[index] # NEVER change this idx to index !!! (debug for 3 hours)
        
        # preprocessing node features obtained from csv file of each subject
        pd_node_ft = pd.read_csv(os.path.join(self.data_dir, subj_name), index_col=0) # set index column with first column
        node_ft = np.array(pd_node_ft.iloc[:self.n_nodes]) # (36, 1) # removing labels in the last, not to be in node features
        # print(node_ft.shape)
        node_ft = node_ft.reshape(-1) # first change shape from (36,1) to (36,): reshape(-1) makes dim = 0: i.e. (n_nodes, )
        # print(node_ft.shape)
        node_ft = node_ft.astype(np.float) # convert type from string to float 
        # print(node_ft)
        # print(subj_name)
        targets = node_ft.astype(int) # and then, convert type from float to int
        # print(targets)
        node_ft_one_hot = np.eye(self.n_feats)[targets] # targets should start from 0
        node_ft_one_hot = torch.from_numpy(node_ft_one_hot) # (n_nodes, n_feats)
        node_ft_one_hot = node_ft_one_hot.type(torch.FloatTensor)
        # print(node_ft_one_hot) # check if generates one-hot vector

        # get group-wise adj mtx obtained from EBICglasso of R
        df_from = pd.read_csv(os.path.join(self.raw_data_dir, f'df_from_MaDE.csv')) # originally 'KAIST'
        df_to = pd.read_csv(os.path.join(self.raw_data_dir, f'df_to_MaDE.csv'))
        df_weight = pd.read_csv(os.path.join(self.raw_data_dir, f'df_weight_MaDE.csv'))
            
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
        
        # make pseudolabel with No, because there's no true labels in unlabeled dataset
        g_No = pd_node_ft.loc['No'].values.astype(float).astype(int).item() # int, 4944
        
        # print(g_No) # added 100000 to SNU No, compared to KAIST, not to be mixed up # 17 -> 100017
        
        label = np.array(g_No, dtype = np.int64) # not [0], but 0 # if dtype is np.uint8, then it only expresses 0~255
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor) # signed int64
        
        self.label = label
        '''
        Here, label means ID (No) of subj, instead of y_true. 
        '''
        
        return self.g, self.label

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
    
#%%
'''
20200417 TODO: 
    1. PHQ-9 in SNU scale: not 0-3 but 1-4 --> fix it !
    2. check for Labelled_SNU.csv, and Labelled_SNU_MINI.csv
    3. make Adj mtx, using Graphical LASSO with EBICglasso in qgraph R package btw items: DO NOT use SHKim's qgraph !
     - tutorial (http://sachaepskamp.com/files/Cookbook.html#network-estimation-binary-data) 
    4. change random_split function to 5-fold CV codes
'''

#%% read refined survey csv files into a single dataframe

def csv2dataframe(questions, labels, single_items, data_dir):
    ''' 
    questions(dict): set questionnaires and specify number of questions    
    labels (list): labels such as YAD_group, suicidal idea 
    single_items(list): single questions such as gender, age, which should be added to item_lst
    data_dir: path to refined survey directories, which contains multiple csv files
    '''
    question_dict = {}
    for key in questions.keys():
        question_lst=[]
        for i in range(questions[key]):
            if key in ['STAI','RAS','RSES']:
                question_lst.append(f'{key}_{i+1:02d}') # STAI_01, not STAI_1
            else:
                question_lst.append(f'{key}_{i+1}')
        question_dict[key]=question_lst
    
    # get new dataframes indexing only questionnaires set above
    item_lst=[]
    for key in question_dict.keys():    
        item_lst.append(question_dict[key])
    
    # add some single items (e.g. labels such as YAD_group, suicidal idea, and single questions such as gender, age)
    item_lst.extend(single_items) # use extend instead of append to unravel elements
    item_lst.extend(labels) # use extend instead of append to unravel elements
    #     item_lst.append('MaDE') # if you use extend for 'MaDE', then you get 'M','a','D', and 'E'.
    
    total_df={} # saving the dataframe before dropout here
    total_df_drop={} # saving the dataframe after dropout here
    for label_key in ['Labelled', 'Unlabelled']: # MUST starts with large charater: i.e. L instead of labelled
        lst_csv = glob.glob(os.path.join(data_dir, f'{label_key}*.csv'))
        # set new dataframes for each label_key: labeled, and unlabeled
        dfs={}           
        new_dfs={}
        new_dfs_drop={}
        for file in lst_csv:
            lst = os.path.basename(file)
            labelness = lst.split('_')[0] # 'Labelled', 'Unlabelled'
            assert labelness == label_key, "wrong label classification"
            
            # get center name from the filename
            center = lst.split('_')[1].split('.')[0] # KAIST, SNU, Gachon, SMC
            
            dfs[center] = pd.read_csv(file, encoding='CP949') # error reading csv without this CP949 d/t 한글
            
            # make new dataframe for only selected items
            new_dfs[center] = pd.DataFrame()
            df_item_lst=[]
            
            for item in item_lst:
                df_item_lst.append(dfs[center][item])
            # only when labeled dataset, then add both MaDE, and YAD_group
            if label_key == 'Labelled':
                df_item_lst.append(dfs[center]['MaDE'])
                df_item_lst.append(dfs[center]['YAD_group'])
            elif label_key == 'Unlabelled':
                dfs[center]['MaDE'] = 0
                dfs[center]['YAD_group'] = 0
                df_item_lst.append(dfs[center]['MaDE'])
                df_item_lst.append(dfs[center]['YAD_group'])
                
            new_dfs[center] = pd.concat(df_item_lst, axis=1)
            
            # delete row with NaN or None: NEVER forget to save the dropna result to another df: i.e. left hand side, new_dfs_drop[center]
            new_dfs_drop[center] = new_dfs[center].dropna(axis=0)
            # print all the items for each df
            print(f'selected item names of {label_key} {center}:', new_dfs[center].columns)
            print(f'dropped NAs: from {new_dfs[center].shape[0]} to {new_dfs_drop[center].shape[0]} in {center}')
        total_df_drop[label_key]=new_dfs_drop
        total_df[label_key]=new_dfs
        
    return total_df_drop, total_df

#%% make dataframe into single csv files

class make_single_csv(object):
    def __init__(self, root_dir, dataframe):
        self.root_dir = root_dir
        self.dataframe = dataframe
        
    def _total_subj2csv(self):
        '''
        Convert subj of dataframe into a single csv in total_data_dir
        '''
        total_data_dir = os.path.join(self.root_dir, 'total_subj_data')
        # remove total_subj_data already made
        if os.path.exists(total_data_dir):
            shutil.rmtree(total_data_dir) # shutil.rmtree enables removing non-empty directories forcefully
        os.makedirs(total_data_dir)
        
        len_label_center={}
        for label_key in self.dataframe.keys():
            print(label_key)
            df_label=self.dataframe[label_key]
            
            for center in df_label.keys():
                print(center)                    
                for subj_idx in range(len(df_label[center])):
                    # saving individual subject dataframe as npy file in data_dir/{label_key}/{center} directory
                    subj_filename = f'{label_key}_{center}_subj{subj_idx}.csv'
                    subj_filepath = os.path.join(total_data_dir, subj_filename)
                    print(total_data_dir)
                    df_label[center].iloc[subj_idx].to_csv(subj_filepath)
                
                if label_key == 'Labelled':
                    len_label_center[center]=len(df_label[center])
                    
        # return len_label_center # CAREFUL indentation !
    
    # make subsampled dataset for class balancing
    def _subsample_balance(self, subsample_ratio = 2):
        '''
        Select labelled and unlabelled data using subsample ratio (i.e. 1:2 -> total 873 subj in data_dir) among 30,935 single csvs in raw_data_dir/total_data_dir
        '''
        total_data_dir = os.path.join(self.root_dir, 'total_subj_data')
        unlabel_lst = [f for f in os.listdir(total_data_dir) if f.startswith('Unlabelled')]
        label_lst = [f for f in os.listdir(total_data_dir) if f.startswith('Labelled')]
        len_label_sum = len(label_lst)
        
        unlabel_center_lst = [f for f in unlabel_lst if f.startswith(f'Unlabelled_KAIST') or f.startswith(f'Unlabelled_SNU')]
        subsample_lst = np.random.choice(len(unlabel_center_lst), int(len_label_sum * subsample_ratio), replace=False) # DO NOT MISS replace option to be FALSE !!
        subsampled_unlabel_center_lst = [unlabel_center_lst[subsample_idx] for subsample_idx in subsample_lst]
        
        # make new data directory for each call
        path, _ = os.path.split(self.root_dir) # root_dir here is actually raw_data directory
        data_dir = os.path.join(path, 'data')  
        if os.path.exists(data_dir):
            print("removing existing data directory")
            shutil.rmtree(data_dir)
        
        print("making new data directory")
        data_dir = os.path.join(path, 'data')   # NEVER erase this line ! or will make an error
        os.mkdir(data_dir)
        
        for filename in subsampled_unlabel_center_lst:
            shutil.copy(os.path.join(total_data_dir, filename), os.path.join(data_dir, filename))
        
        for filename in label_lst:
            shutil.copy(os.path.join(total_data_dir, filename), os.path.join(data_dir, filename))
    
#%% split train, valid, and test idx: implement 5-fold CV here
def split_data(data_dir, split_ratio = 0.8):
    lst_subj = os.listdir(data_dir)
    
    lst_labeled = []
    lst_unlabeled = []
    sum_labeled = 0
    label_dicts = {}
    for idx, subj in enumerate(lst_subj):
        '''
        when split_data is called, you get the index as well as the label of subjects 
        : change code here to change the labels
        '''
        pd_subj=pd.read_csv(os.path.join(data_dir, subj))
        # print(pd_subj.iloc[-2][1])
        # print(pd_subj.loc['MaDE'])
        # if pd_subj['MaDE'] == 1:
        #     label_dicts[idx] = [1]
        #     lst_labeled.append(idx)
        # if pd_subj['MaDE'] == 0:
        #     label_dicts[idx] = [0]
        #     lst_unlabeled.append(idx)
            
        label_dicts[idx] = [int(pd_subj.iloc[-2][1])]
        lst_unlabeled.append(idx)
        
        sum_labeled+=int(pd_subj.iloc[-2][1])
        
        # if subj.startswith('Labelled'):
        #     label_dicts[idx] = [1]
        #     lst_labeled.append(idx)
        # if subj.startswith('Unlabelled'):
        #     label_dicts[idx] = [0]
        #     lst_unlabeled.append(idx)
            
    num_labeled = len(lst_labeled)
    num_unlabeled = len(lst_unlabeled)
    
    train_idx_labeled = np.random.choice(lst_labeled, int(num_labeled * split_ratio), replace=False)
    train_idx_unlabeled = np.random.choice(lst_unlabeled, int(num_unlabeled * split_ratio), replace=False)
    
    train_idx = list(train_idx_labeled) + list(train_idx_unlabeled)
  
    val_test_idx_labeled = list(set(lst_labeled)-set(train_idx_labeled))
    val_test_idx_unlabeled = list(set(lst_unlabeled)-set(train_idx_unlabeled))
    
    valid_idx_labeled = val_test_idx_labeled[:(len(val_test_idx_labeled) // 2)]
    
    valid_idx_unlabeled = val_test_idx_unlabeled[:(len(val_test_idx_unlabeled) // 2)]
    
    valid_idx = valid_idx_labeled + valid_idx_unlabeled   

    test_idx_labeled = val_test_idx_labeled[(len(val_test_idx_labeled) // 2):]
    test_idx_unlabeled = val_test_idx_unlabeled[(len(val_test_idx_unlabeled) // 2):]
    
    test_idx = test_idx_labeled + test_idx_unlabeled   
    
    train_idx.sort() 
    valid_idx.sort()
    test_idx.sort()
    
    partition = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    
    # counting positive/negative cases
    sum_unlabeled = int(len(lst_subj)-sum_labeled)
    print(f'MaDE positive/negative cases are: {sum_labeled}/{sum_unlabeled}')
    return partition, label_dicts

# partition, label_dicts = split_data(data_dir)
#%% split external test set
def sum_labeled_sub(lst_subj, data_dir, label_name):
    sum_labeled=0
    for idx, subj in enumerate(lst_subj):
        # get MaDE label from dataframe per subject
        pd_subj=pd.read_csv(os.path.join(data_dir, subj), index_col=0)
        # print(pd_subj.loc[label_name].values[0]) # 0 or 1
        sum_labeled+=int(float(pd_subj.loc[label_name].values[0]))
        
    return sum_labeled

def split_ext_data(data_dir, ext_center, label_name='MaDE', split_ratio = 0.8):
    lst_subj = os.listdir(data_dir)
    
    print(f'center for external test set:{ext_center}')
    
    lst_ext_subj = [f for f in lst_subj if f.startswith(f'Labelled_{ext_center}')]
    lst_ext_idx = [idx for idx, f in enumerate(lst_subj) if f.startswith(f'Labelled_{ext_center}')]

    print('lst_ext_subj:', len(lst_ext_subj))
    sum_labeledsub = sum_labeled_sub(lst_ext_subj, '../data', label_name)
    print(f'sum of ext:{sum_labeledsub}/{len(lst_ext_subj)}')
    
    sum_labeled = 0
    label_dicts = {}
    
    # first make label_dicts, which collects label per subject using dictionary
    for idx, subj in enumerate(lst_subj):
        
        pd_subj=pd.read_csv(os.path.join(data_dir, subj), index_col=0) # to ignore index and use first column as index column, set index_col = 0
        # get label of specified label name (e.g. MaDE, or suicidal_idea) from dataframe per subject
        label_value = float(pd_subj.loc[label_name].values[0]) # use .values and index [0] to get a scalar from array([1.]), of which shape is [1,]
        # MUST use float first before type convert to int, because '0.0' was initially a string, not a value: https://korbillgates.tistory.com/94
        label_dicts[idx] = [int(label_value)]
        
        sum_labeled+=int(label_value)
        
    lst_train_valid_labeled = [idx for idx, subj in enumerate(lst_subj) if subj.startswith('Labelled_')]
    lst_train_valid_unlabeled = [idx for idx, subj in enumerate(lst_subj) if subj.startswith('Unlabelled_')]

    print('lst_train_valid_labeled:', len(lst_train_valid_labeled))
    print('lst_train_valid_unlabeled:', len(lst_train_valid_unlabeled))

    num_labeled = len(lst_train_valid_labeled)
    num_unlabeled = len(lst_train_valid_unlabeled)
    
    train_idx_labeled = np.random.choice(lst_train_valid_labeled, int(num_labeled * split_ratio), replace=False)
    train_idx_unlabeled = np.random.choice(lst_train_valid_unlabeled, int(num_unlabeled * split_ratio), replace=False)
    
    train_idx = list(train_idx_labeled) + list(train_idx_unlabeled)
  
    valid_idx_labeled = list(set(lst_train_valid_labeled) - set(train_idx_labeled))
    valid_idx_unlabeled = list(set(lst_train_valid_unlabeled) - set(train_idx_unlabeled))
    
    valid_idx = valid_idx_labeled + valid_idx_unlabeled   
    
    test_idx = lst_ext_idx
    
    train_idx.sort() 
    valid_idx.sort()
    test_idx.sort()
    
    partition = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    
    # counting positive/negative cases
    sum_unlabeled = int(len(lst_subj)-sum_labeled)
    print(f'MaDE positive/negative cases are: {sum_labeled}/{sum_unlabeled}')
    
    train_sum_labeled = sum_labeled_sub([lst_subj[f] for f in train_idx], '../data', label_name)
    print(f'train_sum_labeled:{train_sum_labeled}/{len(train_idx)}')
    
    valid_sum_labeled = sum_labeled_sub([lst_subj[f] for f in valid_idx], '../data', label_name)
    print(f'valid_sum_labeled:{valid_sum_labeled}/{len(valid_idx)}')
    
    return partition, label_dicts

# partition, label_dicts = split_ext_data('../data', ext_center)
#%% class Dataset

class survey_dataset(Dataset):
    def __init__(self, data_dir, idx_list, label_dicts, n_nodes, seed, use_random_adj):
        self.data_dir = data_dir
        self.idx_list = idx_list # partition[x]
        self.label_dicts = label_dicts
        self.n_nodes = n_nodes
        self.num_scale = 5
        self.seed = seed
        self.use_random_adj = use_random_adj
        
    def __len__(self):
        return len(self.idx_list) 
    
    def __getitem__(self, index):
        idx = self.idx_list[index] 
        lst = os.listdir(self.data_dir)
        subj_name = lst[idx] # NEVER change this idx to index !!! (debug for 3 hours)
        
        # preprocessing node features obtained from csv file of each subject
        pd_node_ft = pd.read_csv(os.path.join(self.data_dir, subj_name))
        node_ft = np.array(pd_node_ft.iloc[:self.n_nodes,1]) # removing labels in the last, not to be in node features
        node_ft = node_ft[np.newaxis,:].astype(int) # shape: (1,n_nodes)
        targets = np.array(node_ft).reshape(-1)
        node_ft_one_hot = np.eye(self.num_scale)[targets]
        node_ft_one_hot = torch.from_numpy(node_ft_one_hot) # (n_nodes, num_scale)
        node_ft_one_hot = node_ft_one_hot.type(torch.FloatTensor)
        
        # get label from filename of each subject
        label = np.array(self.label_dicts[idx], dtype = np.uint8) # NEVER change this idx to index !!! (debug for 3 hours)  
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        
        # get group-wise adj mtx obtained from EBICglasso of R
        root_dir, _ = os.path.split(self.data_dir) # to use relative path
        raw_data_dir = os.path.join(root_dir, 'raw_data')
        
        # if use_random_adj is True, then get adj matrix randomly for augmentation, chosen between one from KAIST and one from SNU
        if self.use_random_adj:
            np.random.seed(self.seed)
            adj_choice = int(np.random.choice(2, 1))
            lst_unlabeled_centers = ['KAIST','SNU']
            chosen_center = lst_unlabeled_centers[adj_choice]
            df_from = pd.read_csv(os.path.join(raw_data_dir, f'df_from_MaDE.csv')) # {chosen_center}
            df_to = pd.read_csv(os.path.join(raw_data_dir, f'df_to_MaDE.csv'))
            df_weight = pd.read_csv(os.path.join(raw_data_dir, f'df_weight_MaDE.csv'))
        else:
            chosen_center = 'KAIST' # set as default
            df_from = pd.read_csv(os.path.join(raw_data_dir, f'df_from_MaDE.csv'))
            df_to = pd.read_csv(os.path.join(raw_data_dir, f'df_to_MaDE.csv'))
            df_weight = pd.read_csv(os.path.join(raw_data_dir, f'df_weight_MaDE.csv'))
            
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

# def collate(samples):
#     # The input `samples` is a list of pairs (graph, label).
#     graphs, labels = map(list, zip(*samples))
#     for g in graphs:
#         # deal with node feats
#         for key in g.node_attr_schemes().keys():
#             g.ndata[key] = g.ndata[key].float()
#         # no edge feats
#     batched_graph = dgl.batch(graphs)
#     labels = torch.tensor(labels)
#     return batched_graph, labels

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # batched_graph.set_n_initializer(dgl.init.zero_initializer)
    # batched_graph.set_e_initializer(dgl.init.zero_initializer)
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
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
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
        _, pred_class = torch.max(prediction, dim=1)
        pred_class = fn_tonumpy(pred_class)
        label = fn_tonumpy(label)
        print('label:',label)
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

# update best metrics and save the best model for various metrics: loss/sens/spec/acc
def save_best_model(args, val_metrics, epoch, net, optim):
    if args.save_model_metrics == 'loss':
        if val_metrics[args.save_model_metrics] < args.best_metrics:
            print(f'best {args.save_model_metrics}: updated from {args.best_metrics}', end=' ') # remove \n
            args.best_metrics = val_metrics[args.save_model_metrics] # DO NOT save as new_best_metrics, because best_metrics would not be updated from 1e10
            print(f'to {args.best_metrics}')
            save(args, net, optim, epoch)
            args.best_epoch = epoch
        else:
            print(f'not updated {args.save_model_metrics} at epoch {epoch+1} since {args.best_epoch+1}')
    else:
        if val_metrics[args.save_model_metrics] > args.best_metrics:
            print(f'best {args.save_model_metrics}: updated from {args.best_metrics}', end=' ')
            args.best_metrics = val_metrics[args.save_model_metrics]
            print(f'to {args.best_metrics}')
            save(args, net, optim, epoch)
            args.best_epoch = epoch
        else:
            print(f'not updated {args.save_model_metrics} at epoch {epoch+1} since {args.best_epoch+1}')
