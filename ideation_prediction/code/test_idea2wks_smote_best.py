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
# parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--best_epoch', type=int, default=15)
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--ncpu', type=int, default=0)
# parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_best_model', action="store_true", default=False, help='use save_best_model function with various metrics')
parser.add_argument('--save_model_metrics', type=str, default='loss', help='choose among loss/sens/spec/acc')
parser.add_argument('--label_name', type=str, default='suicidal_idea_within_2wk') # 'suicidal_idea'

# directory
parser.add_argument('--new_subsample', action="store_true", default=False, help='subsample new balanced datasets to train')
parser.add_argument('--sample_opt', type=str, default='u2') # undersample: u1, u2 / oversample: SMOTE
# parser.add_argument('--balance_ratio', type=int, default=4) # balance_ratio makes balance of pos:neg = 1:3
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
best_epoch = args.best_epoch
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
args.log_dir = os.path.join(args.log_dir, args.save_model_metrics, args.sample_opt)
log_dir = args.log_dir

# make dir for each sample option: e.g. SMOTE, u1, u2
args.ckpt_dir = os.path.join(args.ckpt_dir, args.save_model_metrics, args.sample_opt)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

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
# get feature names only (except labels)
args.lst_feature = get_feature_columns(questions, single_items, label_lst)[:args.n_nodes]

##### set test dataset
lst_test_filename = ['Unlabelled_SNU.csv'] # _ratio
# args.new_subsample = True # False # 

if args.new_subsample:
    use_split_into_single_csv = True
    
    # set test data directory

    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
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

dataloader = {}
dataloader['test'] = DataLoader(dataset=dataset['test'], batch_size = 640, shuffle=False, num_workers=ncpu, collate_fn = collate, pin_memory = True)#, drop_last=True)

# for class weighted loss
weights = [3.0] # float(args.balance_ratio-2)
class_weights = torch.FloatTensor(weights).to(device)

# define loss, optimizer, and scheduler
loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights)
#%% test

print(f'############ using external test set ############')

test_result_dir = os.path.join(args.result_dir, f'{args.sample_opt}_test')

roc_path = os.path.join(test_result_dir, 'roc_plots')
if not os.path.exists(roc_path):
    os.makedirs(roc_path)

tsne_path = os.path.join(test_result_dir, 'tsne_plots')
if not os.path.exists(tsne_path):
    os.makedirs(tsne_path)

with open(os.path.join(test_result_dir, 'test_metrics.txt'), 'w') as f:
    f.write('loss, sens, spec, acc, new_acc\n')

model = build_model(args)
lst_models = os.listdir(args.ckpt_dir)
lst_models = natsort.natsorted(lst_models) # sorted for filenames with text as well as numbers: e.g. without this line, you get model_epoch9_loss > model_epoch50_loss

for best_epoch in range(len(lst_models)):
    
    ### best_epoch: # 26 for u1 with balance_ratio 4: sens/spec/acc = 75/83/83 # 26 for u2 with balance_ratio 20: sens/spec/acc = 85/73/73
    path_best_model = os.path.join(args.ckpt_dir, lst_models[best_epoch])
    print(f'loading the best {lst_models[best_epoch]} model of {args.sample_opt} model')
    state_dict=torch.load(path_best_model)['net']
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        lst_final_concat = []
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
            
            model_prediction = model(bg, feats) # torch.Size([640]) -- batch_size, not scalar
            prediction = torch.sigmoid(model_prediction)
            final_concat = model.final_concat
            # print(f'prediction score of model: {prediction}')
                
            test_loss = loss_func(prediction, label)
            test_loss_arr += [test_loss.item()]
            
            pred_class = (prediction>0.5).float()
                    
            test_total += label.size(0)
            test_correct += torch.sum(pred_class == label).item()
            # print(f'pred_class:{pred_class}')
            # print(f'label:{label}')
            print(f'test_correct:{test_correct}')
            
            label_np = fn_tonumpy(label)
            prediction_np = fn_tonumpy(prediction)
            pred_class_np = fn_tonumpy(pred_class)
            final_concat_np = fn_tonumpy(final_concat)
            
            lst_test_label.extend(label_np)
            lst_test_prediction.extend(prediction_np)
            lst_test_pred_class.extend(pred_class_np)
            lst_final_concat.extend(final_concat_np)
            
            tn_batch, fp_batch, fn_batch, tp_batch = evaluate(model, bg, feats, label)
            # tn_batch, fp_batch, fn_batch, tp_batch = confusion_matrix(label_np, pred_class_np).ravel()
            # print('batch tn, fp, fn, tp:', tn_batch, fp_batch, fn_batch, tp_batch, 'total:', tn_batch+fp_batch+fn_batch+tp_batch)  
            
            tn+=tn_batch
            fp+=fp_batch
            fn+=fn_batch
            tp+=tp_batch
        
        test_acc = test_correct/test_total * 100
        
        print('tn, fp, fn, tp:', tn, fp, fn, tp, 'total:', test_total)  
        test_sens = tp/(tp+fn) * 100
        test_spec = tn/(tn+fp) * 100        
        test_new_acc = (tp+tn)/(tn+fp+fn+tp) * 100
                        
        # draw roc plot
        fpr, tpr, _ = roc_curve(lst_test_label, lst_test_prediction) 
        roc_auc = auc(fpr, tpr)
        
        # get CI of AUC
        confidence_lower, confidence_upper = CI_for_AUC(roc_path, lst_test_label, lst_test_prediction)  
        
        print(f'TEST of {args.sample_opt} model {best_epoch}|Loss:{np.mean(test_loss_arr)}|sens:{test_sens}|spec:{test_spec}|acc:{test_acc}|new_acc:{test_new_acc}| auc:{roc_auc}({confidence_lower} - {confidence_upper})')
        
        # draw t-sne plots
        plt.ioff()
        plt.figure(figsize=(15,10))
        plt.title(f"plotting t-SNE for {args.label_name}")
        draw_tsne_plot(args, lst_final_concat, lst_test_label, '{args.label_name}') # lst_prediction                    
        print('Saving tsne plot for test')
        plt.savefig(os.path.join(tsne_path, f'tsne_plot_for_test_e{best_epoch+1}.png'), dpi=300)
        plt.close()
        
        # for saving plots for each fold
        plt.ioff()                
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                  lw=2, label=f'ROC curve for test set (AUC = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for test set')
        plt.legend(loc="lower right")
        print(f'Saving ROC plot for test set of {args.sample_opt} model {best_epoch}')
        plt.savefig(os.path.join(roc_path, f'roc_plot_for_test_{args.sample_opt}_model{best_epoch}.png'), dpi=300)
        # plt.show()
        plt.close()
        
        with open(os.path.join(test_result_dir, 'test_metrics.txt'), 'a') as f:
            f.write(f'TEST of {args.sample_opt} model {best_epoch}, Loss:{np.mean(test_loss_arr):.4f}, sens:{test_sens:.4f}, spec:{test_spec:.4f}, acc:{test_acc:.4f}, new_acc:{test_new_acc:.4f}, auc:{roc_auc:.4f}({confidence_lower:.4f} - {confidence_upper:.4f})\n')
        
    