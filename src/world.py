'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

from parse import parse_args

args = parse_args()

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['layers'] = args.layers
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_K'] = args.top_K
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['seed'] = args.seed
config['test_ratio'] = args.test_ratio
config['u_sim'] = args.u_sim
config['i_sim'] = args.i_sim
config['edge'] = args.edge
config['i_K'] = args.i_K
config['u_K'] = args.u_K
config['self_loop'] = bool(args.self_loop)
config['e_attr_mode'] = args.e_attr_mode
config['shuffle'] = args.shuffle
config['neg_sampling'] = args.neg_sampling
config['full_sample'] = args.full_sample
config['n_neg_samples'] = args.n_neg_samples
config['save_sim_mat'] = args.save_sim_mat
config['save_res'] = args.save_res
config['save_pred'] = args.save_pred
config['margin'] = args.margin
config['load'] = args.load