import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from i_sim import jaccard_sim

# Create a dense binary matrix 5 users and 6 items, each row represents a user's interaction with items
dense_matrix = np.array([
    [1, 0, 0, 1, 0, 1],  # user 0
    [0, 0, 0, 1, 1, 0],  # user 1
    [0, 1, 1, 0, 1, 1],  # user 2
    [1, 0, 0, 1, 0, 0],  # user 3
    [0, 1, 0, 1, 1, 0],  # user 4
    [0, 1, 0, 1, 1, 0],  # user 5
    [1, 1, 0, 1, 1, 1],  # user 6
    [1, 0, 0, 0, 0, 0],  # user 7
])

# Convert the dense binary matrix to a sparse matrix (CSR format)
sparse_matrix = csr_matrix(dense_matrix)

# print("Sparse Matrix (CSR format):")
# print(sparse_matrix)
# Convert the sparse matrix to a binary sparse matrix (ensuring no non-binary values)
sparse_matrix.data = (sparse_matrix.data > 0).astype(int)

# Compute sparse cosine similarity between users
u_similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
u_jaccard_similarity_matrix = jaccard_sim(sparse_matrix)

# Print results
print("User Cosine Similarity Matrix (CSR format):")
print(u_similarity_matrix)

print("User Jaccard Similarity Matrix (CSR format):")
print(u_jaccard_similarity_matrix[0])

# Convert back to dense format for better visualization
u_dense_similarity = u_similarity_matrix.toarray()
#print("\nDense Cosine Similarity Matrix:")
#print(dense_similarity)


# # Compute sparse cosine similarity between items
# i_similarity_matrix = cosine_similarity(sparse_matrix.T, dense_output=False)

# # Print results
# print("Item Similarity Matrix (CSR format):")
# print(i_similarity_matrix)

# # Convert back to dense format for better visualization
# i_dense_similarity = i_similarity_matrix.toarray()
# #print("\nDense Cosine Similarity Matrix:")
# #print(dense_similarity)

