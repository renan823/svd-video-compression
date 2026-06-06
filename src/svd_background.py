import numpy as np


def Jacobi(A, tol=1e-6, max_iter=10000):
    n = A.shape[0]
    A_k = np.copy(A)
    V = np.eye(n)

    for _ in range(max_iter):
        p, q = np.unravel_index(np.argmax(np.abs(A_k - np.diag(np.diag(A_k)))), A_k.shape)
        max_item = A_k[p, q]

        if max_item < tol:
            break

        theta = 0.0
        diff = A_k[q, q] - A_k[p, p]
        if abs(A_k[p, q]) < abs(diff) * 1e-36:
            theta = 0.5 * np.pi / 2
        else:
            theta = 0.5 * np.arctan2(2 * A_k[p, q], diff) 

        c = np.cos(theta)
        s = np.sin(theta)

        J = np.eye(n)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s

        A_k = J.T @ A_k @ J
        V = V @ J

    return np.diag(A_k), V


def SVD(M):

    A = M.T @ M

    eigenValues, V = Jacobi(A)#np.linalg.eigh(A)

    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    V = V[:, idx]

    singularValues = np.sqrt(np.maximum(eigenValues, 0))
    sigma = np.diag(singularValues)

    U = M @ V @ np.linalg.pinv(sigma)

    return U, sigma, V.T

def reconstruct_Background(U, sigma, V, k):
    
    Uk = U[:,:k]
    sigmak = sigma[:k, :k]
    Vk = V[:k, :]

    return Uk @ sigmak @ Vk

def error(M, L):
    return np.linalg.norm(M.astype(np.float32) - L.astype(np.float32), ord='fro')

def cumulative_variance(sigma):
    singular_values = np.diag(sigma)**2
    return np.cumsum(singular_values)/np.sum(singular_values)
