from ADMM import ADMM
from DatasetLoader import DatasetLoader
from Visualize import Visualize
import numpy as np
import os
path = './Input/input_video.mp4'
dataloader = DatasetLoader(path)
a = ADMM(dataloader)
a.train()
v = Visualize(a)
v.get_results()
