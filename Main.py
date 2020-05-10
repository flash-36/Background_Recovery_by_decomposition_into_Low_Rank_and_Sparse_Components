from ADMM import ADMM
from DatasetLoader import DatasetLoader

path = './Input/input_video.mp4'
dataloader = DatasetLoader(path, keep_color=True, down_sample=True, image_resolution=(360, 640))
a = ADMM(dataloader)
a.train()  # Train and also create output visualizations
