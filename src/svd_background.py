import numpy as np


def Jacobi(A, tol=1e-10, max_iter=10000):
    n = A.shape[0]
    A_k = np.copy(A)
    V = np.eye(n)

    for _ in range(max_iter):
        off_diag = np.triu(np.abs(A_k), 1)
        p, q = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        max_item = A_k[p, q]

        if abs(max_item) < tol:
            break

        tau = (A_k[q, q] - A_k[p, p]) / (2.0 * A_k[p, q])

        t = 0.0
        if np.isclose(tau, 0):
            t = 1.0
        else:
            t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))

        c = 1.0/np.sqrt(1+t**2)
        s = t * c

        app = A_k[p, p]
        aqq = A_k[q, q]
        apq = A_k[p, q]

        A_k[p, p] = app - t*apq
        A_k[q, q] = aqq + t*apq

        A_k[p, q] = 0.0
        A_k[q, p] = 0.0

        for i in range(n):
            if i != p and i != q:

                aip = A_k[i, p]
                aiq = A_k[i, q]

                A_k[i, p] = c*aip - s*aiq
                A_k[p, i] = A_k[i, p]

                A_k[i, q] = s*aip + c*aiq
                A_k[q, i] = A_k[i, q]

        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]

            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq
        
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
