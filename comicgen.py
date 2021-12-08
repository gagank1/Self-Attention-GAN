import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from sagan_models import Generator
from utils import *

class ComicGenerator:
    def __init__(self, model_path):
        self.weights_path = os.path.join(model_path)

        # Generator params:
        self.batch_size = 1 # change to 64 if this doesn't work
        self.imsize = 128
        self.z_dim = 128
        self.g_conv_dim = 64
        
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim)
        self.G = nn.DataParallel(self.G)
        self.G.load_state_dict(torch.load(self.weights_path, map_location='cpu'))
        self.G.train() # set to training mode

    def generateImage(self):
        z = Variable(torch.randn(self.batch_size, self.z_dim), volatile=True)
        fakeim,_,_ = self.G(z)
        denormed = denorm(fakeim.data)[0].mul(255).clamp(0,255).byte().permute(1,2,0)
        return denormed.numpy()

