from src.videobw import VideoBW
from src.svd_background import SVD, reconstruct_Background, error, cumulative_variance
import numpy as np
import matplotlib.pyplot as plt

def main():
    video = VideoBW()
    video.read("stop-motion.mp4")

    M = video.vectorize().astype(np.float64)

    U, S, VT = SVD(M)

    ks = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

    errors = []
    variances = []

    for k in ks:
        L = reconstruct_Background(U, S, VT, k)
        errors.append(error(M, L))
        variances.append(cumulative_variance(S)[k])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ks, errors, 'o-')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('Reconstruction Error')

    plt.subplot(1, 2, 2)
    plt.plot(ks, variances, 'o-')
    plt.xlabel('k')
    plt.ylabel('Cumulative Variance')
    plt.title('Cumulative Variance')

    plt.tight_layout()
    plt.show()

    L = reconstruct_Background(U, S, VT, 1)
    L = np.clip(L, 0, 255)
    L = L.astype(np.uint8)

    video.expand(L)
    video.write(1)

if __name__ == "__main__":
    main()
