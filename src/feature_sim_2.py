import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def normalize_features(features):
    """
    Normalizes features for better similarity calculation.
    
    Args:
        features (torch.Tensor): Feature tensor to be normalized.
        
    Returns:
        np.ndarray: Normalized feature matrix.
    """
    # Convert to NumPy and normalize
    features_np = features.numpy()
    normalized_features = normalize(features_np, axis=1, norm='l2')  # L2 normalization
    return normalized_features

def calculate_item_similarity(item_features):
    """
    Calculates item-item similarity using cosine similarity.

    Args:
        item_features (torch.Tensor): Encoded item feature tensor.

    Returns:
        np.ndarray: Item-item similarity matrix.
    """
    print("Calculating item-item similarity...")
    print("Item features shape:", item_features.shape)
     # Normalize features for meaningful similarity
    normalized_features = normalize_features(item_features)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(normalized_features)

    return similarity_matrix

def encode_item_features(movies_df, title_model_name='all-MiniLM-L6-v2'):
    """
    Encodes movie features including title embeddings, one-hot encoded genres, 
    and optional date information.

    Args:
        movies_df (pd.DataFrame): DataFrame containing movie data with columns 'title', 'genres', and 'release_date'.
        title_model_name (str): Name of the SentenceTransformer model to use for title encoding.

    Returns:
        torch.Tensor: Combined item feature tensor for all movies.
    """
    # One-hot encode genres
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genres = movies_df[genre_columns].values
    genres = torch.from_numpy(genres).to(torch.float)

    # Load pre-trained sentence transformer model
    model = SentenceTransformer(title_model_name)

    # Encode movie titles
    with torch.no_grad():
        titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        titles = titles.cpu()

    # Convert release dates to numerical features (e.g., timestamp)
    if 'release_date' in movies_df.columns:
        release_dates = movies_df['release_date'].apply(
            lambda x: datetime.strptime(x, "%d-%b-%Y").timestamp() if pd.notnull(x) else 0
        ).values
        release_dates = torch.tensor(release_dates, dtype=torch.float).view(-1, 1)
    else:
        release_dates = None
        
    # Normalize genres
    genres = torch.nn.functional.normalize(genres, p=2, dim=1)

    # Normalize titles
    titles = torch.nn.functional.normalize(titles, p=2, dim=1)

    # Normalize release dates (if included)
    if release_dates is not None:
        release_dates = (release_dates - release_dates.mean()) / release_dates.std()
        release_dates = torch.nn.functional.normalize(release_dates, p=2, dim=0)

    # Concatenate features
    if release_dates is not None:
        movie_features = torch.cat([genres, titles, release_dates], dim=-1)
    else:
        movie_features = torch.cat([genres, titles], dim=-1)

    return movie_features

def encode_user_features(users_df, occupation_model_name='all-MiniLM-L6-v2'):
    """
    Encodes user features including numerical features (age), one-hot encoded gender, 
    occupation embeddings, and zip code embeddings.

    Args:
        users_df (pd.DataFrame): DataFrame containing user data with columns 'age', 'gender', 'occupation', and 'zip_code'.
        occupation_model_name (str): Name of the SentenceTransformer model to use for occupation encoding.

    Returns:
        torch.Tensor: Combined user feature tensor for all users.
    """
    # One-hot encode gender ('M' -> [1, 0], 'F' -> [0, 1])
    gender_mapping = {'M': [1, 0], 'F': [0, 1]}
    genders = users_df['gender'].map(gender_mapping).tolist()
    genders = torch.tensor(genders, dtype=torch.float)

    # Normalize age
    ages = torch.tensor(users_df['age'].values, dtype=torch.float).view(-1, 1)
    ages = (ages - ages.mean()) / ages.std()
    ages = torch.nn.functional.normalize(ages, p=2, dim=0)

    # Encode occupation using a SentenceTransformer model
    model = SentenceTransformer(occupation_model_name)
    with torch.no_grad():
        occupations = model.encode(users_df['occupation'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        occupations = occupations.cpu()

    # Normalize occupations
    occupations = torch.nn.functional.normalize(occupations, p=2, dim=1)

    # Encode zip codes numerically
    zip_codes = users_df['zip_code'].apply(lambda x: int(x) if x.isdigit() else 0).values
    zip_codes = torch.tensor(zip_codes, dtype=torch.float).view(-1, 1)
    zip_codes = (zip_codes - zip_codes.mean()) / zip_codes.std()
    zip_codes = torch.nn.functional.normalize(zip_codes, p=2, dim=0)

    # Concatenate features
    user_features = torch.cat([genders, ages, occupations, zip_codes], dim=-1)

    return user_features


def load_item_file2(file_path):
    """
    Load the u.item file and return a DataFrame with processed columns.

    Args:
        file_path (str): Path to the u.item file.

    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns.
    """
    column_names = [
        'movie_id', 'title', 'release_date', 'video_release_date', 'url',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Read the file
    movies_df = pd.read_csv(
        file_path,
        sep='|',
        names=column_names,
        encoding='latin-1',
        engine='python'
    )

    # Drop unnecessary columns (video_release_date and url)
    movies_df = movies_df.drop(columns=['video_release_date', 'url'])

    return movies_df

def load_user_file2(file_path):
    """
    Load the u.user file and return a DataFrame with processed columns.

    Args:
        file_path (str): Path to the u.user file.

    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns.
    """
    # Define column names based on the u.user file format
    column_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    # Read the file
    users_df = pd.read_csv(
        file_path,
        sep='|',  # Separator used in the u.user file
        names=column_names,  # Column names
        encoding='latin-1',  # Encoding used in the file
        engine='python'  # Use Python engine for compatibility
    )

    return users_df

def load_item_file(file_path, item_mapping_path):
    """
    Load the u.item file, map movie IDs to encoded IDs, and return a processed DataFrame.

    Args:
        file_path (str): Path to the u.item file.
        item_mapping_path (str): Path to the item ID mapping file.

    Returns:
        pd.DataFrame: Processed DataFrame with mapped item IDs.
    """
    column_names = [
        'movie_id', 'title', 'release_date', 'video_release_date', 'url',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Load the movies file
    movies_df = pd.read_csv(
        file_path,
        sep='|',
        names=column_names,
        encoding='latin-1',
        engine='python'
    )

    # Drop unnecessary columns
    movies_df = movies_df.drop(columns=['video_release_date', 'url'])

    # Load item ID mapping
    item_mapping = pd.read_csv(item_mapping_path)
    
    # Map original movie IDs to encoded item IDs
    movies_df = movies_df.merge(
        item_mapping,
        how='inner',
        left_on='movie_id',
        right_on='original_item_id'
    ).drop(columns=['movie_id', 'original_item_id']).rename(columns={'encoded_item_id': 'movie_id'})
    
    print("Movies DataFrame:")
    print(movies_df.shape)
    # print item_mapping length
    print("Item Mapping:")
    print(item_mapping.shape)

    return movies_df

def load_user_file(file_path, user_mapping_path):
    """
    Load the u.user file, map user IDs to encoded IDs, and return a processed DataFrame.

    Args:
        file_path (str): Path to the u.user file.
        user_mapping_path (str): Path to the user ID mapping file.

    Returns:
        pd.DataFrame: Processed DataFrame with mapped user IDs.
    """
    # Define column names based on the u.user file format
    column_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    # Load the users file
    users_df = pd.read_csv(
        file_path,
        sep='|',
        names=column_names,
        encoding='latin-1',
        engine='python'
    )

    # Load user ID mapping
    user_mapping = pd.read_csv(user_mapping_path)
    
    # Map original user IDs to encoded user IDs
    users_df = users_df.merge(
        user_mapping,
        how='inner',
        left_on='user_id',
        right_on='original_user_id'
    ).drop(columns=['user_id', 'original_user_id']).rename(columns={'encoded_user_id': 'user_id'})
    
    print("Users DataFrame:")
    print(users_df.shape)

    return users_df


def create_item_sim(movies_path, top_k=10, verbose=False):
    """
    Create an item-item similarity matrix and filter by top_k most similar items.

    Args:
        movies_path (str): Path to the movies data file.
        top_k (int): Number of most similar items to retain for each item.
        verbose (bool): Whether to print debugging information.

    Returns:
        np.ndarray: Filtered item-item similarity matrix.
    """
    # Load the entire movie dataframe into memory:
    movies_df = load_item_file(movies_path, 'data/ml-100k/item_id_mapping.csv')
    if verbose:
        print("Loaded Movies DataFrame:")
        print(movies_df.head())
    
    # Encode item features
    item_features = encode_item_features(movies_df)
    if verbose:
        print("Encoded Item Features Shape:", item_features.shape)
    
    # Calculate item-item similarity matrix
    item_similarity_matrix = calculate_item_similarity(item_features)
    if verbose:
        print("Initial Item-Item Similarity Matrix Shape:", item_similarity_matrix.shape)

    # Filter to retain only the top_k most similar items for each item
    num_items = item_similarity_matrix.shape[0]
    filtered_similarity_matrix = np.zeros_like(item_similarity_matrix)

    for i in range(num_items):
        # Get the indices of the top_k most similar items for the current item
        top_k_indices = np.argsort(item_similarity_matrix[i])[-top_k:]
        
        # Retain only the top_k similarities
        for idx in top_k_indices:
            filtered_similarity_matrix[i, idx] = item_similarity_matrix[i, idx]

    if verbose:
        print("Filtered Item-Item Similarity Matrix Shape:", filtered_similarity_matrix.shape)

    return filtered_similarity_matrix


def create_item_sim2(movies_path, verbose = False):
    # Load the entire movie dataframe into memory:
    movies_df = load_item_file(movies_path, 'data/ml-100k/item_id_mapping.csv')
    if verbose:
        print(movies_df.head())
    item_features = encode_item_features(movies_df)
    
    if verbose:
        print(item_features.shape)
    #print(item_features[0])

    # Calculate item-item similarity matrix
    item_similarity_matrix = calculate_item_similarity(item_features)

    if verbose:
        print("Item-Item Similarity Matrix Shape:", item_similarity_matrix.shape)

        # Example: Print similarity scores for the first item with all others
        print("Similarity scores for the first item:\n", item_similarity_matrix)
    
    return item_similarity_matrix

def create_user_sim(user_path, top_k=10, verbose=False):
    """
    Create a user-user similarity matrix and filter by top_k most similar users.

    Args:
        user_path (str): Path to the user data file.
        top_k (int): Number of most similar users to retain for each user.
        verbose (bool): Whether to print debugging information.

    Returns:
        np.ndarray: Filtered user-user similarity matrix.
    """
    # Load the user data
    user_df = load_user_file(user_path, 'data/ml-100k/user_id_mapping.csv')
    
    if verbose:
        print("Loaded User DataFrame:")
        print(user_df.head())
    
    # Encode user features
    user_features = encode_user_features(user_df)
    
    if verbose:
        print("Encoded User Features Shape:", user_features.shape)
    
    # Calculate user-user similarity matrix
    user_similarity_matrix = calculate_item_similarity(user_features)
    
    if verbose:
        print("Initial User-User Similarity Matrix Shape:", user_similarity_matrix.shape)

    # Filter to retain only the top_k most similar users for each user
    num_users = user_similarity_matrix.shape[0]
    filtered_similarity_matrix = np.zeros_like(user_similarity_matrix)

    for i in range(num_users):
        # Get the indices of the top_k most similar users for the current user
        top_k_indices = np.argsort(user_similarity_matrix[i])[-top_k:]
        
        # Retain only the top_k similarities
        for idx in top_k_indices:
            filtered_similarity_matrix[i, idx] = user_similarity_matrix[i, idx]

    if verbose:
        print("Filtered User-User Similarity Matrix Shape:", filtered_similarity_matrix.shape)

    return filtered_similarity_matrix


def create_user_sim2(user_path, verbose = False):
    # Load the entire movie dataframe into memory:
    user_df = load_user_file(user_path, 'data/ml-100k/user_id_mapping.csv')
    
    if verbose:
        print(user_df.head())
    
    user_features = encode_user_features(user_df)
    
    if verbose:
        print(user_features.shape)
    #print(item_features[0])

    # Calculate item-item similarity matrix
    user_similarity_matrix = calculate_item_similarity(user_features)

    if verbose:
        print("Item-Item Similarity Matrix Shape:", user_similarity_matrix.shape)

        # Example: Print similarity scores for the first item with all others
        print("Similarity scores for the first item:\n", user_similarity_matrix)
    
    return user_similarity_matrix

def count_sim(sim_matrix):
    count = 0
    count_0 = 0
    count_1 = 0
    count_all = 0
    count_minus_1 = 0
    count_plus_1 = 0
    count_minus_one = 0
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            count_all += 1
            if sim_matrix[i][j] > 0 and sim_matrix[i][j] < 1:
                count += 1
            if sim_matrix[i][j] == 0:
                count_0 += 1
            if sim_matrix[i][j] == 1:
                count_1 += 1
            if sim_matrix[i][j] == -1:
                count_minus_one += 1
            if sim_matrix[i][j] > -1 and sim_matrix[i][j] < 0:
                count_minus_1 += 1
            if sim_matrix[i][j] > 1:
                count_plus_1 += 1

    print("Count of values between 0 and 1 in similarity matrix:", count)
    print("Count of values equal to 0 in similarity matrix:", count_0)
    print("Count of values equal to 1 in similarity matrix:", count_1)
    print("Count of all values in similarity matrix:", count_all)
    print("Count of values between -1 and 0 in similarity matrix:", count_minus_1)
    print("Count of values greater than 1 in similarity matrix:", count_plus_1)
    print("Count of values equal to -1 in similarity matrix:", count_minus_one)

    return count, count_0, count_1, count_all, count_minus_1, count_plus_1, count_minus_one


def show_sim(sim_matrix):
    plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Item-Item Similarity Matrix")
    plt.show()

# movies_path = 'data/ml-100k/u.item'

# item_sim = create_item_sim(movies_path)
# count_sim(item_sim)
# show_sim(item_sim)

# users_path = 'data/ml-100k/u.user'

# user_sim = create_user_sim(users_path)
# count_sim(user_sim)
# show_sim(user_sim)


