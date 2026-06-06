import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from src.videobw import VideoBW
from src.svd_background import (
    SVD,
    reconstruct_Background,
    error,
    cumulative_variance
)

logging.getLogger("libav").setLevel(logging.ERROR)


def main():
    # ======================
    # 1. LEITURA DO VÍDEO
    # ======================
    video = VideoBW()
    video.read("bingo.avi")

    threshold = 30

    output_dir = f"outputs_{threshold}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/info.txt", "w") as f:
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Frames: {len(video.frames)}\n")
        f.write(f"Resolution: {video.width}x{video.height}\n")

    M = video.vectorize().astype(np.float32)

    # ======================
    # 2. SVD
    # ======================
    U, S, VT = SVD(M)

    # ======================
    # 3. ERRO + VARIÂNCIA
    # ======================
    ks = [1, 2, 3, 4, 5, 10, 20, 50, 100]

    errors = []
    variances = []

    for k in ks:
        Lk = reconstruct_Background(U, S, VT, k)
        errors.append(error(M, Lk))
        variances.append(cumulative_variance(S)[k])

    # salvar gráfico erro
    plt.figure()
    plt.plot(ks, errors, 'o-')
    plt.title("Reconstruction Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.savefig(f"{output_dir}/error.png")
    plt.close()

    # salvar gráfico variância
    plt.figure()
    plt.plot(ks, variances, 'o-')
    plt.title("Cumulative Variance")
    plt.xlabel("k")
    plt.ylabel("Variance")
    plt.savefig(f"{output_dir}/variance.png")
    plt.close()

    # ======================
    # 4. DECAY SINGULAR VALUES
    # ======================
    sigma_vals = np.diag(S)

    plt.figure()
    plt.plot(sigma_vals)
    plt.title("Decaimento dos valores singulares")
    plt.xlabel("i")
    plt.ylabel("σ_i")
    plt.savefig(f"{output_dir}/singular_values.png")
    plt.close()

    # ======================
    # 5. BACKGROUND + MOVIMENTO
    # ======================
    L = reconstruct_Background(U, S, VT, 1)

    S_mov = np.abs(M - L)

    mask = (S_mov > threshold) * 255
    mask = mask.astype(np.uint8)

    # ======================
    # 6. FRAME COMPARATIVO
    # ======================
    L_uint8 = np.clip(L, 0, 255).astype(np.uint8)

    video.expand(L_uint8)
    video.write(f"{output_dir}/background.mkv")

    i = 10

    original = M[:, i].reshape(video.height, video.width)
    background = L[:, i].reshape(video.height, video.width)
    movement = mask[:, i].reshape(video.height, video.width)

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Background")
    plt.imshow(background, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Movement")
    plt.imshow(movement, cmap="gray")

    plt.savefig(f"{output_dir}/frames_comparacao.png")
    plt.close()

    # ======================
    # 7. VÍDEO DO MOVIMENTO
    # ======================
    S_vis = np.clip(S_mov * 3, 0, 255).astype(np.uint8)
    video.frames = [
        S_vis[:, i].reshape(video.height, video.width)
        for i in range(S_vis.shape[1])
    ]

    video.write(f"{output_dir}/movement.mkv")

    # ======================
    # 8. COMPARAÇÃO SVD
    # ======================
    U2, s2, VT2 = np.linalg.svd(M, full_matrices=False)

    L_lib = U2[:, :1] @ np.diag(s2[:1]) @ VT2[:1, :]

    print("Erro SVD manual:", error(M, reconstruct_Background(U, S, VT, 1)))
    print("Erro SVD numpy:", error(M, L_lib))


if __name__ == "__main__":
    main()