import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from sagan_models import Generator
from utils import *

class ComicGenerator:
    def __init__(self, model_path, gen_num):
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
        self.gen_num = gen_num

    def generateImage(self):
        z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        fakeim,_,_ = self.G(z)
        denormed = denorm(fakeim.data)

        try:
            os.remove('fakeim'+str(self.gen_num)+'.png') # delete old generated image
        except OSError:
            pass

        save_image(denormed, os.path.join('fakeim'+str(self.gen_num)+'.png'))

