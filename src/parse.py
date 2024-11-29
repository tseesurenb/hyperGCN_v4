'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="tempLGCN", description="Dynamic GCN-based CF recommender")
    parser.add_argument('--model', type=str, default='hyperGCN', help='rec-model, support [LightGCN, NGCF, hyperGCN]')
    parser.add_argument('--dataset', type=str, default='lastfm', help="available datasets: [lastfm, yelp2018, amazon-book]")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of GCN")
    parser.add_argument('--batch_size', type=int, default= 1024, help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int,default=51)
    parser.add_argument('--epochs_per_eval', type=int,default=10)
    parser.add_argument('--verbose', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--scale', type=float, default=1, help="amplify the similarity score")
    parser.add_argument('--decay', type=float, default=1e-04, help="the weight decay for l2 normalizaton")
    parser.add_argument('--top_K', type=int, default=20, help="@k test list")
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--u_sim', type=str, default='cos', help='options are cos (cosine) and jac (jaccard)')
    parser.add_argument('--i_sim', type=str, default='cos', help='options are cos (cosine) and jac (jaccard)')
    parser.add_argument('--edge', type=str, default='knn')
    parser.add_argument('--u_K', type=int, default=20)
    parser.add_argument('--i_K', type=int, default=20)
    parser.add_argument('--self_loop', type=bool, default=False)
    parser.add_argument('--e_attr_mode', type=str, default='exp')
    parser.add_argument('--e_attr_drop', type=float, default=0.0)
    parser.add_argument('--drop_mode', type=float, default=0.0)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--weighted_neg_sampling', type=bool, default=False)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--save_res', type=bool, default=False)
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_sim_mat', type=bool, default=False)
    parser.add_argument('--margin', type=float, default=0.0, help="the margin in BPR loss")
    
    return parser.parse_args()