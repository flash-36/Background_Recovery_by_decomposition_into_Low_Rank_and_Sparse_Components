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


def visualize(L, S, image_resolution, lambd, ini_method, iter, color):
    os.makedirs(f'./Output/lambda={lambd}_initialization={ini_method}', exist_ok=True)

    if iter == -1:
        output_video = cv2.VideoWriter(
            f'./Output/lambda={lambd}_initialization={ini_method}/final_background_foreground.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (2 * image_resolution[1], image_resolution[0]), color)
    else:
        output_video = cv2.VideoWriter(
            f'./Output/lambda={lambd}_initialization={ini_method}/iteration_{iter}_background_foreground.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (2 * image_resolution[1], image_resolution[0]), color)
    for i in range(L.shape[1]):
        back_img_vector = (L[:, i] * 255).astype(np.uint8)
        fore_img_vector = (S[:, i] * 255).astype(np.uint8)
        back_img = np.reshape(back_img_vector, image_resolution)
        fore_img = np.reshape(fore_img_vector, image_resolution)
        img = np.concatenate((back_img, fore_img), axis=1)
        output_video.write(img)
    output_video.release()
