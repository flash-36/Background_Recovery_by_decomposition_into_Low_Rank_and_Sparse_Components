from ADMM import ADMM
from DatasetLoader import DatasetLoader

path = './Input/input_video.mp4'
dataloader = DatasetLoader(path)
a = ADMM(dataloader)
a.train() # Train and also create output visualizations

