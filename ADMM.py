from DatasetLoader import DatasetLoader
from Utils import *
import matplotlib.pyplot as plt
class ADMM:
    def __init__(self, path):
        dataloader = DatasetLoader(path,200)
        self.video_matrix = dataloader.get_matrix()
        #Hyperparameters
        self.rho = 3
        self.lambd = 2
        self.max_iter = 10000
        self.threshold = 10e-3

    def train(self):
        #Random initializations
        L = np.random.random(self.video_matrix.shape)
        S = np.random.random(self.video_matrix.shape)
        A = np.random.random(self.video_matrix.shape)
        objective_values=[]
        obj = objective(L,S,self.lambd)
        objective_values.append(obj)
        for iter in range(self.max_iter):
            L = SVT((self.video_matrix-S-A/self.rho),1/self.rho)
            S = ST((self.video_matrix-L-A/self.rho),self.lambd/self.rho)
            A = A + self.rho*(L+S-self.video_matrix)

            obj = objective(L,S,self.lambd)
            objective_values.append(obj)

            diff = np.abs(objective_values[-1]-objective_values[-2])
            if diff<self.threshold:
                break
            if iter%10==0:
                print(obj)
        else:
            print("Didn't converge")

        plt.plot(objective_values)
        plt.show()
        return L,S






