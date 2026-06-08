import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from src.movement import detect_movement
from src.videobw import VideoBW
from src.svd import (
    SVD,
    reconstruct_background,
    error,
    cumulative_variance,
)


# Logger do codec (só ignorar logs chatos)
logging.getLogger("libav").setLevel(logging.ERROR)


'''
Salva resultados de erro e variância 
no formato de gráfico.
Exibe na stdout os resultados numéricos.
'''
def save_variances(dir: str, ks: list[int], errors: list[float], variances: list[float]):
    # Valores calculados do erro e da variancia
    print(f"\n{'k':>5} {'Error':>12} {'Variance':>12}")
    print("-" * 31)
    
    for k, e, v in zip(ks, errors, variances):
        print(f"{k:>5} {e:>12.4f} {v:>12.4f}")
    print()

    # Salvar gráfico erro
    plt.figure()
    plt.plot(ks, errors, 'o-')
    plt.title("Reconstruction Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.savefig(f"{dir}/error.png")
    plt.close()

    # Salvar gráfico variância
    plt.figure()
    plt.plot(ks, variances, 'o-')
    plt.title("Cumulative Variance")
    plt.xlabel("k")
    plt.ylabel("Variance")
    plt.savefig(f"{dir}/variance.png")
    plt.close()


'''
Salva resultados do decaimento
dos valores singulares como um gráfico.
'''
def save_decay(S: np.ndarray, dir: str):
    sigma_vals = np.diag(S)

    plt.figure()
    plt.plot(sigma_vals)
    plt.title("Decaimento dos valores singulares")
    plt.xlabel("i")
    plt.ylabel("σ_i")
    plt.savefig(f"{dir}/singular_values.png")
    plt.close()


'''
Função principal.
'''
def main():
    # Leitura do vídeo em bw
    video = VideoBW()
    video.read("videos/stop-motion.mp4")

    threshold = 30

    # Diretório de resultados
    output_dir = f"outputs_{threshold}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/info.txt", "w") as f:
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Frames: {len(video.frames)}\n")
        f.write(f"Resolution: {video.width}x{video.height}\n")

    # Matriz dos frames
    M = video.vectorize().astype(np.float32)

    # Aplicação do SVD
    U, S, VT = SVD(M)

    # Erros e variâncias para reconstrução
    ks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    errors = []
    variances = []

    for k in ks:
        Lk = reconstruct_background(U, S, VT, k)
        errors.append(error(M, Lk))
        variances.append(cumulative_variance(S)[k])

    # Salvar valores e decaimento
    save_variances(output_dir, ks, errors, variances)
    save_decay(S, output_dir)

    # Detecção de movimento
    detect_movement(M, U, S, VT, threshold, output_dir)

    # Comparação SVD numpy
    U2, s2, VT2 = np.linalg.svd(M, full_matrices=False)

    L_lib = U2[:, :1] @ np.diag(s2[:1]) @ VT2[:1, :]

    print("Erro SVD manual:", error(M, reconstruct_background(U, S, VT, 1)))
    print("Erro SVD numpy:", error(M, L_lib))


if __name__ == "__main__":
    main()