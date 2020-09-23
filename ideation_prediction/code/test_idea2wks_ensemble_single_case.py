import os
import glob
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
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
# parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--ncpu', type=int, default=0)
# parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_best_model', action="store_true", default=False, help='use save_best_model function with various metrics')
parser.add_argument('--save_model_metrics', type=str, default='loss', help='choose among loss/sens/spec/acc')
parser.add_argument('--label_name', type=str, default='suicidal_idea_within_2wk') # 'suicidal_idea'

# directory
parser.add_argument('--new_subsample', action="store_true", default=False, help='subsample new balanced datasets to train')
parser.add_argument('--sample_opt', type=str, default='single') # undersample: u1, u2 / oversample: SMOTE
# parser.add_argument('--balance_ratio', type=int, default=20) # balance_ratio makes balance of pos:neg = 1:3
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

# batch_size = args.batch_size
seed = args.seed
lr = args.lr
ncpu = args.ncpu
# num_epochs = args.num_epochs
args.best_epoch = 0

args.smote_ratio = 1.0 # 0.7

### set directory using thr
args.data_dir += "_thr" + args.thr
args.log_dir += "_thr" + args.thr
args.ckpt_dir += "_thr" + args.thr
args.result_dir += "_thr" + args.thr

result_dir = args.result_dir

raw_data_dir = args.raw_data_dir
args.data_dir = os.path.join(args.data_dir, f'{args.sample_opt}')

# define train, valid, and test directory
train_data_dir = os.path.join(args.data_dir, 'train')
valid_data_dir = os.path.join(args.data_dir, 'valid')
test_data_dir = os.path.join(args.data_dir, 'single_test')
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

# make a result directory
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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

##### set test dataset
lst_test_filename = ['single_case_MaDE_pseudo.csv'] # _ratio
args.new_subsample = True # False # 

if args.new_subsample:
    use_split_into_single_csv = True
    
    # set test data directory
    
    if not os.path.exists(test_data_dir):
        # shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)   
    
    df_test = make_datasets(args, questions, single_items, label_lst, lst_test_filename).df_total_drop
    df_test = convert_total_quantile(args, df_test, dataset=None) # set dataset as None: using quantile_values obtained from train dataset
    df_test.to_csv(os.path.join(args.csv_dir, 'df_test_single_case.csv'), index=False)
    
    if use_split_into_single_csv:
        split_into_single_csv(df_test, test_data_dir)
    else:
        print("not choose to split into single csv files")        
        
dataset = {}
dataset['test'] = survey_dataset(args, test_data_dir)

dataloader = {}
dataloader['test'] = DataLoader(dataset=dataset['test'], batch_size = 1, shuffle=False, num_workers=ncpu, collate_fn = collate, pin_memory = True)#, drop_last=True)

# for class weighted loss
weights = [3.0] # float(args.balance_ratio-2)
class_weights = torch.FloatTensor(weights).to(device)

# define loss, optimizer, and scheduler
loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights)

#%%
print(f'############ using external test set ############')

lst_sample_opt = ['u1','u2','SMOTE']
lst_best_epoch = [23,17,12] # [33,14,12] # [21,21,1] # # set best epoch for each model

# best metrics:
# 1. best = [21,13,18] with metrics 78.03/77.71/77.71/0.852(0.826-0.876)
# 2. best = [21,32,18] with metrics 76.30/81.54/81.47/0.855(0.831-0.878)
# 3. best = [21,26,1] with metrics 79.77/77.85/77.87/0.869(0.846-0.891)

lst_model_weight = [1.0, 1.0, 1.0] # best
# print(f'lst_model_weight for without dgi model:{lst_model_weight}')

num_of_models = sum(lst_model_weight)
    
ensemble_result_dir = os.path.join(args.result_dir, 'single')

pred_classnames = ['non-ideator','suicidal ideator']

lst_best_model = []

for best_epoch, sample_opt in zip(lst_best_epoch, lst_sample_opt):

    model = build_model(args)
    # make dir for each sample option: e.g. SMOTE, u1, u2
    sample_opt_path = os.path.join(args.ckpt_dir, args.save_model_metrics, sample_opt)
    lst_models = os.listdir(sample_opt_path)
    lst_models = natsort.natsorted(lst_models) # sorted for filenames with text as well as numbers: e.g. without this line, you get model_epoch9_loss > model_epoch50_loss
      
    path_best_model = os.path.join(sample_opt_path, lst_models[best_epoch])
    # print(f'loading the {best_epoch}th {sample_opt} model for ensemble')
    # print(f'loading the {lst_models[best_epoch]} model of {sample_opt} model')
    
    state_dict=torch.load(path_best_model)['net']
    model.load_state_dict(state_dict)
    model = model.to(device)
    lst_best_model.append(model)

with torch.no_grad():
    lst_best_model[0].eval()
    lst_best_model[1].eval()
    lst_best_model[2].eval()
    
    lst_test_label = []
    lst_test_prediction = []
    lst_test_pred_class = []
    test_loss_arr=[]
    test_correct=0
    test_total=0
    
    tn, fp, fn, tp = 0, 0, 0, 0

    for batch, (bg, label) in enumerate(dataloader['test']):
        
        feats = bg.ndata.pop('n')
        feats, label = feats.to(device), label.to(device)
                    
        print('########## using ensemble ########')
        sum_model_prediction = torch.zeros([len(label)]).to(device) # not batch_size, but len(label): because of 608 not 640 case
        sum_model_prediction.requires_grad = True
        
        for model_idx, model_weight in enumerate(lst_model_weight): # range(num_of_models):
            
            ln_prediction = lst_best_model[model_idx](bg, feats) # torch.Size([640]) -- batch_size, not scalar
            model_prediction = torch.sigmoid(ln_prediction)
            # print(f'prediction score of model {lst_sample_opt[model_idx]}: {model_prediction}')
            sum_model_prediction += model_prediction * model_weight
        prediction = sum_model_prediction / num_of_models
        # print(f'prediction score of ensemble model: {prediction.item()*100:.2f} %')
        
        test_loss = loss_func(prediction, label)
        test_loss_arr += [test_loss.item()]
        
        pred_class = (prediction>0.5).float()
        pred_class_idx = int(pred_class.item())
    
    true_pred_score = [pred_class_idx, int(label.item()), prediction.item()*100]
    # print(f'predicted class:{pred_classnames[pred_class_idx]}')
    # print(f'true class:{pred_classnames[int(label.item())]}')
    
    # save the true and pred class to load for saliency plot
    with open(os.path.join(ensemble_result_dir, 'single_pred_true_score.pkl'), 'wb') as f:
        pickle.dump(true_pred_score, f)
        