import numpy as np


'''
Função que implementa o método de Jacobi
para cálculo dos autovalores e autovetores.
'''
def jacobi_method(A: np.ndarray, tol=1e-10, max_iter=10000):
    n = A.shape[0]

    # Ak tem autovalores na diagonal
    # V tem os autovetores
    A_k = np.copy(A)
    V = np.eye(n)

    for _ in range(max_iter):
        # Maior módulo fora da diagonal
        off_diag = np.triu(np.abs(A_k), 1)
        p, q = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        max_item = A_k[p, q]

        if abs(max_item) < tol:
            break

        # Variáveis auxiliares para calcular C e S
        tau = (A_k[q, q] - A_k[p, p]) / (2.0 * A_k[p, q])

        t = 1.0
        if not np.isclose(tau, 0):
            t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))

        # Valores de C e S (cos e sen)
        c = 1.0/np.sqrt(1+t**2)
        s = t * c

        # Indices pp e qq
        A_k[p, p] -= t * A_k[p, q]
        A_k[q, q] += t * A_k[p, q]

        # Indices pq e qp zerados
        A_k[p, q] = 0.0
        A_k[q, p] = 0.0

        # Indices ip, iq e pj, qj
        for i in range(n):
            if i != p and i != q:
                aip = A_k[i, p]
                aiq = A_k[i, q]

                A_k[i, p] = c*aip - s*aiq
                A_k[p, i] = A_k[i, p]

                A_k[i, q] = s*aip + c*aiq
                A_k[q, i] = A_k[i, q]

        # Multiplicação das colunas
        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]

            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq
        
    return np.diag(A_k), V


'''
Cálculo do SVD usando método de Jacobi.
'''
def SVD(M: np.ndarray):
    A = M.T @ M

    # Autoval e Autovec
    eigval, V = jacobi_method(A)

    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    V = V[:, idx]

    singularValues = np.sqrt(np.maximum(eigval, 0))
    sigma = np.diag(singularValues)

    U = M @ V @ np.linalg.pinv(sigma)

    return U, sigma, V.T


'''
Função de reconstrução do fundo do vídeo
utilizando os valores do SVD.
'''
def reconstruct_background(U: np.ndarray, sigma: np.ndarray, V: np.ndarray, k: int):
    Uk = U[:,:k]
    sigmak = sigma[:k, :k]
    Vk = V[:k, :]

    return Uk @ sigmak @ Vk


'''
Função para o cálculo do erro usando 
a norma de Frobenius.
'''
def error(M: np.ndarray, L: np.ndarray):
    return np.linalg.norm(M.astype(np.float32) - L.astype(np.float32), ord='fro')


'''
Função para o cálculo da variância cumulativa.
'''
def cumulative_variance(sigma: np.ndarray):
    singular_values = np.diag(sigma) ** 2
    return np.cumsum(singular_values) / np.sum(singular_values)
