from DatasetLoader import DatasetLoader
import os
from Utils import *
import matplotlib.pyplot as plt


class ADMM:
    def __init__(self, dataloader):
        self.keep_color = dataloader.keep_color
        self.video_matrix = dataloader.get_matrix()
        self.image_resolution = dataloader.image_resolution

        # Hyperparameters
        self.rho = 0.3
        self.lambd_min = 0.0002
        self.lambd_max = 0.002
        self.max_iter = 10000
        self.threshold = 0.2  # Relative threshold percentage
        self.ini_method = ['random', 'video_matrix_semi', 'video_matrix_full']

        self.L = None
        self.S = None

    def train(self):

        color = 0 if not self.keep_color else 1
        for ini_method in self.ini_method:
            for lambd in np.linspace(self.lambd_min, self.lambd_max, 1):

                lambd = round(lambd, 5)

                if ini_method == 'random':
                    # Random initializations
                    L = np.random.random(self.video_matrix.shape)
                    S = np.random.random(self.video_matrix.shape)
                    A = np.random.random(self.video_matrix.shape)

                elif ini_method == 'video_matrix_semi':
                    # Initialize to video_matrix M
                    L = self.video_matrix
                    S = np.random.random(self.video_matrix.shape)
                    A = np.random.random(self.video_matrix.shape)

                elif ini_method == 'video_matrix_full':
                    # Initialize to video_matrix M
                    L = self.video_matrix
                    S = self.video_matrix
                    A = np.random.random(self.video_matrix.shape)
                else:
                    raise NotImplementedError

                objective_values = []
                obj = objective(L, S, lambd)
                objective_values.append(obj)

                for iter in range(self.max_iter):
                    L = SVT((self.video_matrix - S - A / self.rho), 1 / self.rho)
                    S = ST((self.video_matrix - L - A / self.rho), lambd / self.rho)
                    A = A + self.rho * (L + S - self.video_matrix)

                    obj = objective(L, S, lambd)
                    objective_values.append(obj)

                    diff = np.abs((objective_values[-1] - objective_values[-2]) / objective_values[-2]) * 100
                    if diff < self.threshold:
                        break

                    print(f'lambda : {lambd}     iteration : {iter}     objective value : {obj}')
                    if iter % 20 == 0:
                        visualize(L, S, self.image_resolution, lambd, ini_method, iter, color)



                else:
                    print("Didn't converge")

                visualize(L, S, self.image_resolution, lambd, ini_method, -1, color)
                plt.plot(objective_values)
                plt.title(f'Training Curve, lambda={lambd}')
                plt.xlabel('no. of iterations')
                plt.ylabel('objective value')
                plt.savefig(f'./Output/lambda={lambd}_initialization={ini_method}/Training_Curve_{ini_method}.png')

        self.S = S
        self.L = L
