import numpy as np
import cv2
import os
from pywt import threshold


def SVT(X, tau):
    U, S, V = np.linalg.svd(X, full_matrices=False)
    T = np.maximum(0, S - tau)
    T = np.diagflat(T)
    A = np.matmul(U, T)
    return np.matmul(A, V)


def ST(X, tau):
    return threshold(X, tau, 'soft')


def objective(L, S, lambd):
    return np.linalg.norm(L, ord='nuc') + lambd * np.linalg.norm(S, ord=1)


def visualize(L, S, image_resolution, lambd, iter):
    os.makedirs(f'./Output/lambda={lambd}', exist_ok=True)
    height, width, layers = image_resolution
    if iter == -1:
        output_video = cv2.VideoWriter(f'./Output/lambda={lambd}/final_background_foreground.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   10, (2*width, height))
    else:
        output_video = cv2.VideoWriter(f'./Output/lambda={lambd}/iteration_{iter}_background_foreground.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   10, (2*width, height))
    for i in range(L.shape[1]):
        back_img_vector = (L[:, i] * 255).astype(np.uint8)
        fore_img_vector = (S[:, i] * 255).astype(np.uint8)
        back_img = np.reshape(back_img_vector, image_resolution)
        fore_img = np.reshape(fore_img_vector, image_resolution)
        img = np.concatenate((back_img,fore_img),axis=1)
        output_video.write(img)
    output_video.release()


if __name__ == '__main__':
    A = np.linspace(-4, 4, 7)
    print(A)
    print(ST(A, 2.5))
