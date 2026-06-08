import numpy as np
import matplotlib.pyplot as plt

from src.svd import reconstruct_background
from src.videobw import VideoBW


'''
Função para detecção do movimento
no vídeo, usanod a diferença entre fundo e
demais objetos na cena.
'''
def detect_movement(
    M: np.ndarray, 
    U: np.ndarray, 
    S: np.ndarray, 
    VT: np.ndarray, 
    threshold: int, 
    dir: str,
):
    # Detecção de movimento
    L = reconstruct_background(U, S, VT, 1)
    
    S_mov = np.abs(M - L)
    mask = (S_mov > threshold) * 255
    mask = mask.astype(np.uint8)

    # Video do background
    video = VideoBW()
    video.expand(np.clip(L, 0, 255).astype(np.uint8))
    video.write(f"{dir}/background.mkv")

    i = 10

    # Extração do movimento
    original = M[:, i].reshape(video.height, video.width)
    background = L[:, i].reshape(video.height, video.width)
    movement = np.clip(S_mov[:, i] * 3, 0, 255).reshape(video.height, video.width)

    # Plot das informações de movimento
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Background")
    plt.imshow(background, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Movement")
    plt.imshow(movement, cmap="gray")

    plt.savefig(f"{dir}/frames_comparacao.png")
    plt.close()

    # Gerar frames do movimento
    S_vis = np.clip(S_mov * 3, 0, 255).astype(np.uint8)
    video.frames = [
        S_vis[:, i].reshape(video.height, video.width)
        for i in range(S_vis.shape[1])
    ]

    # Vídeo do movimento
    video.write(f"{dir}/movement.mkv")