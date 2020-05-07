import cv2
import numpy as np


class DatasetLoader:
    def __init__(self, path, frame_limit=10000):
        self.path = path
        self.frame_limit = frame_limit



    def get_matrix(self,keep_color=False):
        cap = cv2.VideoCapture(self.path)
        success, matrix = cap.read()
        if not keep_color:
            matrix = cv2.cvtColor(matrix,cv2.COLOR_BGR2GRAY)
        matrix = np.ravel(matrix)
        frame_no = 1
        while success and frame_no < self.frame_limit:
            success, img = cap.read()
            if not keep_color:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = np.ravel(img)
            frame_no += 1
            matrix = np.vstack([matrix, img])
        return matrix.transpose()

if __name__ == '__main__':
    path = './Input/input_video.mp4'
    data_loader = DatasetLoader(path,20)
    gray_matrix = data_loader.get_matrix()
    print(gray_matrix.shape)
    color_matrix = data_loader.get_matrix(True)
    print(color_matrix.shape)
