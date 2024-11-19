import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Create a dense binary matrix
dense_matrix = np.array([
    [1, 0, 1, 0, 1],  # Row 0
    [0, 1, 1, 0, 0],  # Row 1
    [1, 0, 1, 0, 1],  # Row 2 (same as Row 0)
    [0, 0, 0, 1, 1],  # Row 3
    [1, 1, 1, 0, 0],  # Row 4
])

# Convert the dense binary matrix to a sparse matrix (CSR format)
sparse_matrix = csr_matrix(dense_matrix)

# Convert the sparse matrix to a binary sparse matrix (ensuring no non-binary values)
sparse_matrix.data = (sparse_matrix.data > 0).astype(int)

# Compute sparse cosine similarity
similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)

# Print results
print("Sparse Cosine Similarity Matrix (CSR format):")
print(similarity_matrix)

# Convert back to dense format for better visualization
dense_similarity = similarity_matrix.toarray()
print("\nDense Cosine Similarity Matrix:")
print(dense_similarity)
