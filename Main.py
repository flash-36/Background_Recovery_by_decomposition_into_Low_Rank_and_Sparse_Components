from ADMM import ADMM
path = './Input/input_video.mp4'
a = ADMM(path)
L,S = a.train()

