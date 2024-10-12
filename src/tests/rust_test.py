import numpy as np
#from neg_uniform_sample_module import neg_uniform_sample
from rust_utils import sum_as_string, neg_uniform_sample


# Example Data
train_df = np.array([[0, 1], [1, 2], [2, 3]])  # Example training data (user, positive item)

# Convert full_adj_list to a dictionary
full_adj_list = {
    0: {"neg_items": [4, 5, 6]},  # User 0
    1: {"neg_items": [7, 8, 9]},  # User 1
    2: {"neg_items": [10, 11, 12]},  # User 2
}


n_usr = 100  # Example offset for positive and negative items

# Call the Rust function
S = sum_as_string(2, 3)
print(S)

S = neg_uniform_sample(train_df, full_adj_list, n_usr)
print(S)
