'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import torch.backends
import torch.mps
import numpy as np
from procedure import exec_exp
from utils import print_metrics, set_seed, plot_results
import data_prep as dp 
from world import config

# STEP 1: set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# STEP 2: Load the data
train_df, test_df = dp.load_data_from_adj_list(dataset = config['dataset'])

num_users = train_df['user_id'].nunique()
num_items = train_df['item_id'].nunique()
num_interactions = len(train_df) + len(test_df)

dataset_stats = {'num_users': num_users, 'num_items': num_items,  'num_interactions': num_interactions}

# STEP 3: Execute the experiment
#seeds = [2020, 12, 89, 91, 41]
seeds = [2020]
exp_n = 1

recalls, precs, f1s, ncdg, max_indices = [], [], [], [], []
all_losses, all_metrics = [], []

for seed in seeds:
    
    set_seed(seed)
    
    losses, metrics = exec_exp(train_df, test_df, exp_n, seed, device, config['verbose'])
    
    ncdg.append(np.max(metrics['ncdg']))
    max_idx = np.argmax(metrics['ncdg'])
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    
    max_indices.append(max_idx)
    
    all_losses.append(losses)
    all_metrics.append(metrics)
    
    exp_n += 1

print_metrics(recalls, precs, f1s, ncdg, max(max_indices), stats=dataset_stats)

file_name = f"models/{config['model']}_{device}_{config['seed']}_{config['dataset']}_{config['batch_size']}__{config['layers']}_{config['epochs']}_{config['edge']}"
plot_results(file_name, exp_n, all_losses, all_metrics)

