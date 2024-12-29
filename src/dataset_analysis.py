import data_prep as dp


# STEP 1: Load the data
train_df1, test_df1 = dp.load_data_from_adj_list(dataset = 'ml-100k')
train_df2, test_df2 = dp.load_data_from_adj_list(dataset = 'ml-1m')
train_df3, test_df3 = dp.load_data_from_adj_list(dataset = 'gowalla')
train_df4, test_df4 = dp.load_data_from_adj_list(dataset = 'yelp2018')
train_df5, test_df5 = dp.load_data_from_adj_list(dataset = 'amazon_book')

# STEP 2: print sparsity of the dataset which has user and item interactions
num_users = train_df1['user_id'].nunique()
num_items = train_df1['item_id'].nunique()

sparsity1 = 1 - len(train_df1) / (num_users * num_items)

print(f"Sparsity of the dataset: {sparsity1:.3f}")\
    
# STEP 3: show a plot where on x-axis we have users and on y-axis we have number of interactions
# for each user
import pandas as pd
import matplotlib.pyplot as plt

def plot_user_interactions_distribution(train_df):
    """
    Plots the distribution of the number of interactions per user.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing 'user_id' and 'item_id' columns.
    """
    # Calculate the number of interactions per user
    user_interactions = train_df.groupby('user_id').size()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(user_interactions, bins=50, log=False, alpha=0.75)
    plt.title('Number of Users vs. Number of Interactions Distribution')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users (log scale)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
def plot_user_interactions_distributions(train_df1, train_df2, train_df3, train_df4, train_df5, labels):
    """
    Plots the distribution of the number of interactions per user for three datasets side by side.

    Parameters:
    train_df1, train_df2, train_df3 (pd.DataFrame): DataFrames containing 'user_id' and 'item_id' columns.
    labels (list): List of labels for the datasets (e.g., ['Dataset 1', 'Dataset 2', 'Dataset 3']).
    """
    # Calculate the number of interactions per user for each dataset
    user_interactions1 = train_df1.groupby('user_id').size()
    user_interactions2 = train_df2.groupby('user_id').size()
    user_interactions3 = train_df3.groupby('user_id').size()
    user_interactions4 = train_df4.groupby('user_id').size()
    user_interactions5 = train_df5.groupby('user_id').size()

    # Set up the subplots
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)

    datasets = [user_interactions1, user_interactions2, user_interactions3, user_interactions4, user_interactions5]
    for i, (user_interactions, label) in enumerate(zip(datasets, labels)):
        axes[i].hist(user_interactions, bins=50, log=True, alpha=0.75)
        axes[i].set_title(f'Dataset ({label})')
        axes[i].set_xlabel('Number of Interactions')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            axes[i].set_ylabel('Number of Users (log scale)')

    plt.tight_layout()
    plt.show()



def plot_item_interactions_distributions(train_df1, train_df2, train_df3, train_df4, train_df5, labels):
    """
    Plots the distribution of the number of interactions per item for five datasets side by side.

    Parameters:
    train_df1, train_df2, train_df3, train_df4, train_df5 (pd.DataFrame): DataFrames containing 'user_id' and 'item_id' columns.
    labels (list): List of labels for the datasets (e.g., ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5']).
    """
    # Calculate the number of interactions per item for each dataset
    item_interactions1 = train_df1.groupby('item_id').size()
    item_interactions2 = train_df2.groupby('item_id').size()
    item_interactions3 = train_df3.groupby('item_id').size()
    item_interactions4 = train_df4.groupby('item_id').size()
    item_interactions5 = train_df5.groupby('item_id').size()

    # Set up the subplots
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)

    datasets = [item_interactions1, item_interactions2, item_interactions3, item_interactions4, item_interactions5]
    for i, (item_interactions, label) in enumerate(zip(datasets, labels)):
        axes[i].hist(item_interactions, bins=50, log=True, alpha=0.75)
        axes[i].set_title(f'Dataset ({label})')
        axes[i].set_xlabel('Number of Interactions')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            axes[i].set_ylabel('Number of Items (log scale)')

    plt.tight_layout()
    plt.show()

# Example usage
# plot_user_interactions_distributions(train_df1, train_df2, train_df3, ['Dataset 1', 'Dataset 2', 'Dataset 3'])



# Example usage
# Assuming train_df is a DataFrame with 'user_id' and 'item_id' columns
# plot_user_interactions_distribution(train_df)

#plot_user_interactions_distribution(train_df)

#plot_user_interactions_distributions(train_df1, train_df2, train_df3, train_df4, train_df5, ['ml-100k', 'ml-1m', 'gowalla', 'yelp2018', 'amazon_book'])

plot_item_interactions_distributions(train_df1, train_df2, train_df3, train_df4, train_df5, ['ml-100k', 'ml-1m', 'gowalla', 'yelp2018', 'amazon_book'])