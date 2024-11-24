import torch
import torch.backends
import torch.mps
import utils as ut
import data_prep as dp 
from world import config

from world import config
from data_prep import get_edge_index, create_uuii_adjmat

device = "cuda" if torch.cuda.is_available() else "cpu"

# STEP 2: Load the data
orig_train_df, orig_test_df = dp.load_data_from_adj_list(dataset = config['dataset'])

# _test_df = orig_test_df[
#       (orig_test_df['user_id'].isin(orig_train_df['user_id'].unique())) & \
#       (orig_test_df['item_id'].isin(orig_train_df['item_id'].unique()))
#     ]

_train_df, _test_df = ut.encode_ids(orig_train_df, orig_test_df)
    
N_USERS = _train_df['user_id'].nunique()
N_ITEMS = _train_df['item_id'].nunique()

    
adj_list = ut.make_adj_list(_train_df) # adj_list is a user dictionary with a list of positive items (pos_items) and negative items (neg_items)
        
if config['edge'] == 'knn': # edge from a k-nearest neighbor or similarity graph
    
    knn_train_adj_df, item_sim_dict = create_uuii_adjmat(_train_df) 
    knn_edge_index, knn_edge_attrs = get_edge_index(knn_train_adj_df)
    knn_edge_index = torch.tensor(knn_edge_index).to(device).long()
                
    edge_index = knn_edge_index.to(device)
    edge_attrs = torch.tensor(knn_edge_attrs).to(device)