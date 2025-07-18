
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from scipy import stats
import config

def compute_rdm(X, metric='correlation'):
    """Computes the representational dissimilarity matrix (RDM)."""
    distances = pdist(X, metric=metric)
    return squareform(distances)

def reduce_dimensionality(X, method='pca', n_components=10):
    """Performs dimensionality reduction on neural data."""
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'mds':
        reducer = MDS(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'isomap':
        reducer = Isomap(n_components=n_components)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    embedding = reducer.fit_transform(X)
    return embedding, reducer

def correlate_geometry(embedding, behavioral_vars):
    """Correlates behavioral variables with neural geometry."""
    results = {}
    for var_name, var_values in behavioral_vars.items():
        correlations = [stats.pearsonr(embedding[:, dim], var_values)[0] for dim in range(embedding.shape[1])]
        results[var_name] = {'correlations': np.array(correlations)}
    return results

def compare_conditions(embedding, condition_labels, n_permutations=1000):
    """Compares neural embeddings between conditions."""
    
    def centroid_distance(data, labels):
        unique_labels = np.unique(labels)
        centroids = [np.mean(data[labels == l], axis=0) for l in unique_labels]
        return np.linalg.norm(centroids[0] - centroids[1])

    observed_dist = centroid_distance(embedding, condition_labels)
    
    permuted_dists = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(condition_labels)
        permuted_dists.append(centroid_distance(embedding, permuted_labels))
        
    p_value = (np.sum(np.array(permuted_dists) >= observed_dist) + 1) / (n_permutations + 1)
    
    return {'observed_distance': observed_dist, 'p_value': p_value} 