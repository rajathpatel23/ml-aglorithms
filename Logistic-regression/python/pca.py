import numpy as np

def pca(X, num_components):
    # Centering the data
    X_meaned = X - np.mean(X, axis=0)
    
    # Computing covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)
    
    # Eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    
    # Sorting eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    # Selecting a subset from the rearranged eigenvalue matrix
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    
    # Transforming the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    return X_reduced

# Example usage
X = np.random.rand(100, 5)  # 100 samples, 5 features
num_components = 2
X_reduced = pca(X, num_components)

print(X_reduced.shape)  # Should be (100, 2)