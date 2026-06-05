import numpy as np

def SVD(M):

    A = M.T @ M

    eigenValues, V = np.linalg.eigh(A)

    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    V = V[:, idx]

    singularValues = np.sqrt(np.maximum(eigenValues, 0))
    sigma = np.diag(singularValues)

    U = M @ V @ np.linalg.inv(sigma)

    return U, sigma, V.T

def reconstruct_Background(U, sigma, V, k):
    
    Uk = U[:,:k]
    sigmak = sigma[:k, :k]
    Vk = V[:k, :]

    return Uk @ sigmak @ Vk

def error(M, L):
    return np.linalg.norm(M.astype(np.float64) - L.astype(np.float64), ord='fro')

def cumulative_variance(sigma):
    singular_values = np.diag(sigma)**2
    return np.cumsum(singular_values)/np.sum(singular_values)
