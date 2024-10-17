'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import os
import torch
import numpy as np
import torch.nn.functional as F
import utils as ut

from tqdm import tqdm
from model import RecSysGNN
from world import config
from data_prep import get_edge_index, create_uuii_adjmat
import time
import sys

#from rust_utils import neg_uniform_sample

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"
 
def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0, margin = 0.5):
    
    margin = config['margin']
    
    if config['n_neg_samples'] == 1:
        neg_reg_loss = neg_emb0.norm(2).pow(2)
    else:
        neg_reg_loss = neg_emb0.norm(2).pow(2).sum() / neg_emb0.shape[1]  # Sum over negatives and average by N
        
    # Compute regularization loss
    reg_loss = (1 / 2) * (
        user_emb0.norm(2).pow(2) + 
        pos_emb0.norm(2).pow(2)  +
        neg_reg_loss
    ) / float(len(users))
    
    # Compute positive and negative scores
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)  # [batch_size]
    
    if config['n_neg_samples'] == 1:
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores) + margin)  # Using softplus for stability
    else:
        # Neg scores for each user and N negative items: [batch_size, N]
        neg_scores = torch.mul(users_emb.unsqueeze(1), neg_emb).sum(dim=2)
        #mbpr_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(1))))
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores.unsqueeze(1) + margin))  # Using softplus for stability
        
    return bpr_loss, reg_loss

def train_and_eval(model, optimizer, train_df, test_df, edge_index, edge_attrs, adj_list, item_sim_dict, device, exp_n, g_seed):
   
    epochs = config['epochs']
    b_size = config['batch_size']
    topK = config['top_k']
    decay = config['decay']
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    losses = { 'bpr_loss': [], 'reg_loss': [], 'total_loss': [] }
    metrics = { 'recall': [], 'precision': [], 'f1': [], 'ncdg': [] }
    
    max_ncdg = 0.0
    max_epoch = 0
    
    #train_array = train_df.to_numpy()
    neg_sample_time = 0.0
    
    pbar = tqdm(range(epochs), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    
    for epoch in pbar:
    
        total_losses, bpr_losses, reg_losses  = [], [], []
        
        # Shuffle the DataFrame
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        if config['n_neg_samples'] == 1:
            #S = neg_uniform_sample(train_array, train_neg_adj_list, n_users)
            S = ut.neg_uniform_sample(train_df, adj_list, item_sim_dict, n_users)
        else:
            S = ut.multiple_neg_uniform_sample(train_df, adj_list, n_users)
         
        users = torch.Tensor(S[:, 0]).long().to(device)
        pos_items = torch.Tensor(S[:, 1]).long().to(device)
        neg_items = torch.Tensor(S[:, 2]).long().to(device)
                
        if config['shuffle']: 
            users, pos_items, neg_items = ut.shuffle(users, pos_items, neg_items)
        
        n_batches = len(users) // b_size + 1
        
        if epoch % config["epochs_per_eval"] == 0:
            model.eval()
            with torch.no_grad():
                _, out = model(edge_index, edge_attrs)
                final_u_emb, final_i_emb = torch.split(out, (n_users, n_items))
                recall,  prec, ncdg = ut.get_metrics(final_u_emb, final_i_emb, n_users, n_items, train_df, test_df, topK, device)
            
            f1 = (2 * recall * prec / (recall + prec)) if (recall + prec) != 0 else 0.0
                
            #losses['bpr_loss'].append(round(np.mean(bpr_losses),4))
            #losses['reg_loss'].append(round(np.mean(reg_losses),4))
            #losses['total_loss'].append(round(np.mean(total_losses),4))
            
            losses['bpr_loss'].append(round(np.mean(bpr_losses), 4) if bpr_losses else np.nan)
            losses['reg_loss'].append(round(np.mean(reg_losses), 4) if reg_losses else np.nan)
            losses['total_loss'].append(round(np.mean(total_losses), 4) if total_losses else np.nan)
            
            metrics['recall'].append(round(recall,4))
            metrics['precision'].append(round(prec,4))
            metrics['f1'].append(round(f1,4))
            metrics['ncdg'].append(round(ncdg,4))
            
            if ncdg > max_ncdg:
                max_ncdg = ncdg
                max_epoch = epoch
            
            pbar.set_postfix_str(f"prec {br}{prec:.4f}{rs} | recall {br}{recall:.4f}{rs} | ncdg {br}{ncdg:.4f} ({max_ncdg:.4f} at {max_epoch}) {rs}")
            pbar.refresh()
                                
        model.train()
        for (b_i, (b_users, b_pos, b_neg)) in enumerate(ut.minibatch(users, pos_items, neg_items, batch_size=b_size)):
                                     
            u_emb, pos_emb, neg_emb, u_emb0,  pos_emb0, neg_emb0 = model.encode_minibatch(b_users, b_pos, b_neg, edge_index, edge_attrs)
            bpr_loss, reg_loss = compute_bpr_loss(b_users, u_emb, pos_emb, neg_emb, u_emb0,  pos_emb0, neg_emb0)
            
            reg_loss = decay * reg_loss
            total_loss = bpr_loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bpr_losses.append(bpr_loss.item())
            reg_losses.append(reg_loss.item())
            total_losses.append(total_loss.item())
            
            # Update the description of the outer progress bar with batch information
            pbar.set_description(f"{config['model']}({g_seed:2}) | #ed {len(edge_index[0]):6} | ep({epochs}) {epoch} | ba({n_batches}) {b_i:3} | n_sample_t({neg_sample_time:.2}) | loss {total_loss.item():.4f}")
            
    return (losses, metrics)

def exec_exp(orig_train_df, orig_test_df, exp_n = 1, g_seed=42, device='cpu', verbose = -1):
    
    _test_df = orig_test_df[
      (orig_test_df['user_id'].isin(orig_train_df['user_id'].unique())) & \
      (orig_test_df['item_id'].isin(orig_train_df['item_id'].unique()))
    ]

    _train_df, _test_df = ut.encode_ids(orig_train_df, _test_df)
        
    N_USERS = _train_df['user_id'].nunique()
    N_ITEMS = _train_df['item_id'].nunique()
    
    if verbose >= 0:
        print(f"dataset: {br}{config['dataset']} {rs}| seed: {g_seed} | exp: {exp_n} | device: {device}")
        print(f"{br}Trainset{rs} | #users: {N_USERS}, #items: {N_ITEMS}, #interactions: {len(_train_df)}")
        print(f" {br}Testset{rs} | #users: {_test_df['user_id'].nunique()}, #items: {_test_df['item_id'].nunique()}, #interactions: {len(_test_df)}")
      
    adj_list = ut.make_adj_list(_train_df) # adj_list is a user dictionary with a list of positive items (pos_items) and negative items (neg_items)
     
    if config['edge'] == 'bi': # edge from a bipartite graph
        
        u_t = torch.LongTensor(_train_df.user_id)
        i_t = torch.LongTensor(_train_df.item_id) + N_USERS
    
        bi_edge_index = torch.stack((
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        )).to(device)
        
        edge_index = bi_edge_index.to(device)
        edge_attrs = None
        
        item_sim_dict = None
         
    if config['edge'] == 'knn': # edge from a k-nearest neighbor or similarity graph
        
        knn_train_adj_df, item_sim_dict = create_uuii_adjmat(_train_df, verbose) 
        knn_edge_index, knn_edge_attrs = get_edge_index(knn_train_adj_df)
        knn_edge_index = torch.tensor(knn_edge_index).to(device).long()
                    
        edge_index = knn_edge_index.to(device)
        edge_attrs = torch.tensor(knn_edge_attrs).to(device)
    
    cf_model = RecSysGNN(model=config['model'], emb_dim=config['emb_dim'],  n_layers=config['layers'], n_users=N_USERS, n_items=N_ITEMS, edge_attr_mode = config['e_attr_mode'], self_loop=config['self_loop']).to(device)
    opt = torch.optim.Adam(cf_model.parameters(), lr=config['lr'])
    
    model_file_path = f"./models/params/{config['model']}_{device}_{g_seed}_{config['dataset']}_{config['batch_size']}__{config['layers']}_{config['epochs']}_{config['edge']}"
    
    if config['load'] and os.path.exists(model_file_path):
        cf_model.load_state_dict(torch.load(model_file_path, weights_only=True))

    losses, metrics = train_and_eval(cf_model, 
                                     opt, 
                                     _train_df,
                                     _test_df, 
                                     edge_index, 
                                     edge_attrs,
                                     adj_list,
                                     item_sim_dict,
                                     device,
                                     exp_n, 
                                     g_seed)
   

    # Assume 'model' is your PyTorch model
    torch.save(cf_model.state_dict(), model_file_path)

    return losses, metrics