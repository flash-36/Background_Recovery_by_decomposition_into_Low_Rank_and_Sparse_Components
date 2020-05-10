from DatasetLoader import DatasetLoader
import os
from Utils import *
import matplotlib.pyplot as plt
class ADMM:
    def __init__(self, dataloader):
        self.video_matrix = dataloader.get_matrix()
        self.image_resolution = dataloader.image_resolution

        # Hyperparameters
        self.rho = 0.3
        self.lambd_min = 0.0002
        self.lambd_max = 0.002
        self.max_iter = 10000
        self.threshold = 0.2  # Relative threshold percentage


        self.L = None
        self.S = None

    def train(self):
        for lambd in np.linspace(self.lambd_min,self.lambd_max,10):

            lambd = round(lambd,5)

            # Random initializations
            L = np.random.random(self.video_matrix.shape)
            S = np.random.random(self.video_matrix.shape)
            A = np.random.random(self.video_matrix.shape)
            objective_values=[]
            obj = objective(L,S,lambd)
            objective_values.append(obj)

            for iter in range(self.max_iter):
                L = SVT((self.video_matrix-S-A/self.rho),1/self.rho)
                S = ST((self.video_matrix-L-A/self.rho),lambd/self.rho)
                A = A + self.rho*(L+S-self.video_matrix)

                obj = objective(L,S,lambd)
                objective_values.append(obj)

                diff = np.abs((objective_values[-1]-objective_values[-2])/objective_values[-2])*100
                if diff<self.threshold:
                    break

                print(f'lambda : {lambd}     iteration : {iter}     objective value : {obj}')
                if iter%20==0:
                    visualize(L,S,self.image_resolution,lambd,iter)



            else:
                print("Didn't converge")

            visualize(L,S,self.image_resolution,lambd,-1)
            plt.plot(objective_values)
            plt.title(f'Training Curve, lambda={lambd}')
            plt.xlabel('no. of iterations')
            plt.ylabel('objective value')
            plt.savefig(f'./Output/lambda={lambd}/Training_Curve.png')


        self.S = S
        self.L = L






