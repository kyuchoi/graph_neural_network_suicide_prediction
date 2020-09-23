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
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
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
parser.add_argument('--sample_opt', type=str, default='u2') # undersample: u1, u2 / oversample: SMOTE
parser.add_argument('--thr', type=str, default='6') # string for diff thr from R: 4,5,6 for thr 0.4, 0.5, 0.6
# parser.add_argument('--balance_ratio', type=int, default=20) # balance_ratio makes balance of pos:neg = 1:3
# parser.add_argument('--split_ratio', type=float, default=0.1) # split_ratio makes train:valid = 9:1
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

# making seperate log and checkpoint directory for each setting: e.g. save_model_metrics
# args.log_dir = os.path.join(args.log_dir, args.save_model_metrics, args.sample_opt)
# log_dir = args.log_dir

# if not os.path.exists(args.ckpt_dir):
#     os.makedirs(args.ckpt_dir)

result_dir = args.result_dir

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
args.lst_feature_label = get_feature_columns(questions, single_items, label_lst)
# get feature names only (except labels)
args.lst_feature = args.lst_feature_label[:args.n_nodes]

##### set test dataset
lst_test_filename = ['Unlabelled_SNU.csv'] # _ratio
# args.new_subsample = True # False # 

if args.new_subsample:
    use_split_into_single_csv = True
    
    # set test data directory

    if not os.path.exists(test_data_dir):
        # shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)   
    
    df_test = make_datasets(args, questions, single_items, label_lst, lst_test_filename).df_total_drop
    df_test = convert_total_quantile(df_test)
    df_test.to_csv(os.path.join(args.csv_dir, 'df_test.csv'), index=False)
     
    if use_split_into_single_csv:
        split_into_single_csv(df_test, test_data_dir)
    else:
        print("not choose to split into single csv files")      

dataset = {}
dataset['test'] = survey_dataset(args, test_data_dir)

lst_test_len = len(dataset['test']) # os.listdir(test_data_dir) # 13408
sequential_sampler = BatchSampler(SequentialSampler(range(lst_test_len)), batch_size=640, drop_last=False)#(dataset['test'])
# print(list(sequential_sampler['test']))

dataloader = {}
dataloader['test'] = DataLoader(dataset=dataset['test'], batch_sampler = sequential_sampler, shuffle=False, num_workers=ncpu, collate_fn = collate, pin_memory = True)#, drop_last=True)

# for class weighted loss
weights = [3.0] # float(args.balance_ratio-2)
class_weights = torch.FloatTensor(weights).to(device)

# define loss, optimizer, and scheduler
loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights)

#%%
print(f'############ using external test set ############')

lst_sample_opt = ['SMOTE','u1','u2']
lst_best_epoch = [1,33,-1] # new best perf: [1,21,26] # [1,33,-1] # [1,34,24] # set best epoch for each model
lst_model_weight = [1.0, 1.0, 1.0] # best
# print(f'lst_model_weight for without dgi model:{lst_model_weight}')

num_of_models = sum(lst_model_weight)

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
    model.eval() # MUST use model.eval() here for attention map (debug 3hrs)
    lst_best_model.append(model)

# set path for indiv and avg attention maps
indiv_figpath = os.path.join(args.result_dir, 'individual maps')
if not os.path.exists(indiv_figpath):
    # shutil.rmtree(indiv_figpath)
    os.makedirs(indiv_figpath)
    
avg_figpath = os.path.join(args.result_dir, f'average maps')
if not os.path.exists(avg_figpath):
    # shutil.rmtree(avg_figpath)
    os.makedirs(avg_figpath)

############ NEVER use no_grad() or eval() for attention map (or grad-CAM) (debug 3hrs) #############
# with torch.no_grad():
#     lst_best_model[0].eval()
#     lst_best_model[1].eval()
#     lst_best_model[2].eval()
    
lst_test_label = []
lst_test_prediction = []
lst_test_pred_class = []

test_correct=0
test_total=0

tn, fp, fn, tp = 0, 0, 0, 0

# initialize dict for saliency matrix
saliency_test_dict = {'positive':[], 'negative':[]}

for batch, (bg, label) in enumerate(dataloader['test']):
    
    feats = bg.ndata.pop('n') # ['n']#
    feats, label = feats.to(device), label.to(device)
                
    print('########## using ensemble ########')
    
    num_graphs = int(len(bg)//args.n_nodes)
    print("num_graphs: ", num_graphs)   
    
    for test_num in np.arange(1, num_graphs+1):
    
        # print(f'batch:{batch}, test graph number:{test_num}')
        test_feats = feats[(args.n_nodes * (test_num-1)):(args.n_nodes * test_num),:]
        test_feats.requires_grad_()
        # print(f'test_feats:{test_feats}')
        
        test_label = int(label[test_num-1].cpu().data)
        print(f'{args.label_name} label:{test_label}')
        
        test_bg = dgl.unbatch(bg)[test_num-1]
        # print(test_bg.ndata['n']) # works only when you set line 223 as follows: feats = bg.ndata['n'], instead of popping feats = bg.ndata.pop('n')
        
        # prediction = lst_best_model[1](test_bg, test_feats)
        # pred_score = torch.sigmoid(prediction)
        # print(f'pred_score:{pred_score * 100}')
        # prediction.requires_grad_()
        
        # test_pred_score, test_pred_class = torch.max(test_sigmoid_prediction.unsqueeze(0), dim=1) # removed .unsqueeze(0) here only in server, but not in local PC !!: but don't know why 
        sum_model_prediction = 0
        
        for model_idx, model_weight in enumerate(lst_model_weight): # range(num_of_models):
            
            ln_prediction = lst_best_model[model_idx](test_bg, test_feats) # torch.Size([640]) -- batch_size, not scalar
            model_prediction = torch.sigmoid(ln_prediction)
            print(f'prediction score of model {lst_sample_opt[model_idx]}: {model_prediction}')
            sum_model_prediction += model_prediction * model_weight
        pred_score = sum_model_prediction / num_of_models
        pred_score.requires_grad_()
        print(f'pred_score (%):{pred_score * 100:0.02f}')
    
        pred_class = (pred_score>0.5).float()
                            
        pred_score.backward() # because grad can be implicitly created only for scalar outputs: error when test_prediction.backward()
        saliency_test = test_feats.grad.data.cpu() 
        # print('saliency_test:', saliency_test) # print saliency matrix
        
        len_batch = len(dataloader['test'])
        print(f'batch:{batch}/{len_batch}')
        print(f'test_num:{test_num}/{num_graphs}')
        
        
        # NOT draw individual saliency for saving time
        draw_indiv_saliency = False        
        if draw_indiv_saliency:
            
            plt.ioff()
                   
            # normalize the heatmap
            # print("test graph number: ", test_num)
            # print("before norm:", saliency_test)
            saliency_test_norm = normalize(saliency_test)#, axis=0, norm='l2')
            # print("after norm:", saliency_test_norm)
            # in case of getting saliency per question as well as answer (0~5 point)
            fig = plt.figure(figsize = (12,4))
            plt.imshow(saliency_test.transpose(1,0), cmap='coolwarm') # _norm
            # get node names for xticks
            test_idx = sequential_sampler.batch_size * batch + test_num-1 # dataloader['test'].batch_size
            print(f'test_idx:{test_idx+1}/{lst_test_len+1} th data')
            test_df = pd.read_csv(os.path.join(test_data_dir, os.listdir(test_data_dir)[test_idx]), index_col=0)
            # node_names = list(test_df.index)[:args.n_nodes] 
            node_names = args.lst_feature # _label
            positions = range(len(node_names))
            patient_id = int(test_df.loc['ID'].values)
            print(f'patient_id:{patient_id}')
            plt.title(f'id {patient_id} graph')
            plt.colorbar(label='color')
            
            # display information of the test patient
            plt.gcf().text(0.9, 0.8, f'ID:\n{patient_id}') # plt.gcf() makes the text outside the plot(axes)
            plt.gcf().text(0.9, 0.6, f'true label:\n{test_label}')
            plt.gcf().text(0.9, 0.4, f'pred label:\n{int(pred_class)}')
            plt.gcf().text(0.9, 0.2, f'pred score (%):\n{float(pred_score * 100):0.02f}')
            plt.xticks(positions, node_names, rotation=90)
            plt.yticks(range(args.n_feats))
            plt.savefig(os.path.join(indiv_figpath, f'indiv_saliency_{patient_id}_label{test_label}_pred{pred_class}.png'), dpi=300)
            # plt.show()
            plt.close(fig)
        
        if test_label == 1:
            saliency_test_dict['positive'].append(saliency_test) # _norm
        elif test_label == 0:
            saliency_test_dict['negative'].append(saliency_test) # _norm

#%% 
plot_saliency_maps(args, avg_figpath, saliency_test_dict)

