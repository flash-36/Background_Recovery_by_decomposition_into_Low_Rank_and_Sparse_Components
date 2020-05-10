import cv2
import numpy as np


class DatasetLoader:
    def __init__(self, path, keep_color, down_sample=False, image_resolution=None, frame_limit=600):
        self.path = path
        self.frame_limit = frame_limit
        self.image_resolution = image_resolution
        self.down_sample = down_sample
        self.keep_color = keep_color

    def get_matrix(self):
        print('Reading data...')
        cap = cv2.VideoCapture(self.path)
        success, matrix = cap.read()
        if not self.keep_color:
            matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        if self.down_sample:
            matrix = cv2.resize(matrix, dsize=(self.image_resolution[1], self.image_resolution[0]))

        self.image_resolution = tuple(matrix.shape)
        matrix = np.ravel(matrix)
        frame_no = 1
        while success and frame_no < self.frame_limit:
            success, img = cap.read()
            if not success:
                break
            if not self.keep_color:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.down_sample:
                img = cv2.resize(img, dsize=(self.image_resolution[1], self.image_resolution[0]))
            img = np.ravel(img)
            frame_no += 1
            matrix = np.vstack([matrix, img])
        print('Done')
        return matrix.transpose() / 255
