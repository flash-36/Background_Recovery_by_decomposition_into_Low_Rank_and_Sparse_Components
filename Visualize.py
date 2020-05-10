import cv2
import numpy as np
import matplotlib.pyplot as plt
class Visualize:
    def __init__(self,admm,path='blah'):
        self.path = path
        self.L ,self.S , self.video_matrix, self.image_resolution = admm.L, admm.S, admm.video_matrix, admm.image_resolution
        assert self.L.shape == self.S.shape
        assert self.L.shape == self.video_matrix.shape
    def get_results(self):

        height, width, layers = self.image_resolution
        back_out = cv2.VideoWriter('./Output/background.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
        fore_out = cv2.VideoWriter('./Output/foreground.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
        for i in range(self.video_matrix.shape[1]):
            back_img_vector = (self.L[:,i]*255).astype(np.uint8)
            fore_img_vector = (self.S[:,i]*255).astype(np.uint8)
            back_img = np.reshape(back_img_vector,self.image_resolution)
            fore_img = np.reshape(fore_img_vector,self.image_resolution)
            back_out.write(back_img)
            fore_out.write(fore_img)
        back_out.release()
        fore_out.release()
        plt.imshow(self.video_matrix)
        plt.colorbar()
        plt.show()
        raju = input('raju')
        plt.imshow(self.L)
        plt.colorbar()
        plt.show()
        raju = input('raju')
        plt.imshow(self.S)
        plt.colorbar()
        plt.show()
        raju = input('raju')