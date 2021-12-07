import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

# FIXES FROM: https://github.com/heykeetae/Self-Attention-GAN/issues/12

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.zeros_like(x).normal_(std=self.std)
        else:
            return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    # def __init__(self,in_dim,activation):
    #     super(Self_Attn,self).__init__()
    #     self.chanel_in = in_dim
    #     self.activation = activation
        
    #     self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
    #     self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
    #     self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
    #     self.gamma = nn.Parameter(torch.zeros(1))

    #     self.softmax  = nn.Softmax(dim=-1) #

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    # def forward(self,x):
    #     """
    #         inputs :
    #             x : input feature maps( B X C X W X H)
    #         returns :
    #             out : self attention value + input feature 
    #             attention: B X N X N (N is Width*Height)
    #     """
    #     m_batchsize,C,width ,height = x.size()
    #     proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
    #     proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
    #     energy =  torch.bmm(proj_query,proj_key) # transpose check
    #     attention = self.softmax(energy) # BX (N) X (N) 
    #     proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

    #     out = torch.bmm(proj_value,attention.permute(0,2,1) )
    #     out = out.view(m_batchsize,C,width,height)
        
    #     out = self.gamma*out + x
    #     return out,attention

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #print('attention size {}'.format(x.size()))
        m_batchsize, C, width, height = x.size()
        #print('query_conv size {}'.format(self.query_conv(x).size()))
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    """Generator."""

    # def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
    #     super(Generator, self).__init__()
    #     self.imsize = image_size
    #     layer1 = []
    #     layer2 = []
    #     layer3 = []
    #     last = []

    #     repeat_num = int(np.log2(self.imsize)) - 3
    #     mult = 2 ** repeat_num # 8
    #     layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
    #     layer1.append(nn.BatchNorm2d(conv_dim * mult))
    #     layer1.append(nn.ReLU())

    #     curr_dim = conv_dim * mult

    #     layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
    #     layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
    #     layer2.append(nn.ReLU())

    #     curr_dim = int(curr_dim / 2)

    #     layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
    #     layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
    #     layer3.append(nn.ReLU())

    #     if self.imsize == 64:
    #         layer4 = []
    #         curr_dim = int(curr_dim / 2)
    #         layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
    #         layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
    #         layer4.append(nn.ReLU())
    #         self.l4 = nn.Sequential(*layer4)
    #         curr_dim = int(curr_dim / 2)

    #     self.l1 = nn.Sequential(*layer1)
    #     self.l2 = nn.Sequential(*layer2)
    #     self.l3 = nn.Sequential(*layer3)

    #     last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
    #     last.append(nn.Tanh())
    #     self.last = nn.Sequential(*last)

    #     self.attn1 = Self_Attn( 128, 'relu')
    #     self.attn2 = Self_Attn( 64,  'relu')


    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, attn_feat=[16, 32], upsample=False):
        super(Generator, self).__init__()
        self.imsize = image_size
        layers = []

        n_layers = int(np.log2(self.imsize)) - 2
        mult = 8 #2 ** repeat_num  # 8
        assert mult * conv_dim > 3 * (2 ** n_layers), 'Need to add higher conv_dim, too many layers'

        curr_dim = conv_dim * mult

        # Initialize the first layer because it is different than the others.
        layers.append(SpectralNorm(nn.ConvTranspose2d(z_dim, curr_dim, 4)))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU())

        for n in range(n_layers - 1):
            layers.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layers.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layers.append(nn.ReLU())

            #check the size of the feature space and add attention. (n+2) is used for indexing purposes
            if 2**(n+2) in attn_feat:
                layers.append(Self_Attn(int(curr_dim / 2), 'relu'))
            curr_dim = int(curr_dim / 2)

        # append a final layer to change to 3 channels and add Tanh activation
        layers.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        layers.append(nn.Tanh())

        self.output = nn.Sequential(*layers)

    # def forward(self, z):
    #     z = z.view(z.size(0), z.size(1), 1, 1)
    #     out=self.l1(z)
    #     out=self.l2(out)
    #     out=self.l3(out)
    #     out,p1 = self.attn1(out)
    #     out=self.l4(out)
    #     out,p2 = self.attn2(out)
    #     out=self.last(out)

    #     return out, p1, p2

    def forward(self, z):
        #TODO add dynamic layers to the class for inspection. if this is done we can output p1 and p2, right now they
        # are a placeholder so training loop can be the same.
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.output(z)
        p1 = []
        p2 = []
        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    # def __init__(self, batch_size=64, image_size=64, conv_dim=64):
    #     super(Discriminator, self).__init__()
    #     self.imsize = image_size
    #     layer1 = []
    #     layer2 = []
    #     layer3 = []
    #     last = []

    #     layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
    #     layer1.append(nn.LeakyReLU(0.1))

    #     curr_dim = conv_dim

    #     layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
    #     layer2.append(nn.LeakyReLU(0.1))
    #     curr_dim = curr_dim * 2

    #     layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
    #     layer3.append(nn.LeakyReLU(0.1))
    #     curr_dim = curr_dim * 2

    #     if self.imsize == 64:
    #         layer4 = []
    #         layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
    #         layer4.append(nn.LeakyReLU(0.1))
    #         self.l4 = nn.Sequential(*layer4)
    #         curr_dim = curr_dim*2
    #     self.l1 = nn.Sequential(*layer1)
    #     self.l2 = nn.Sequential(*layer2)
    #     self.l3 = nn.Sequential(*layer3)

    #     last.append(nn.Conv2d(curr_dim, 1, 4))
    #     self.last = nn.Sequential(*last)

    #     self.attn1 = Self_Attn(256, 'relu')
    #     self.attn2 = Self_Attn(512, 'relu')

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, attn_feat=[16, 32]):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layers = []

        n_layers = int(np.log2(self.imsize)) - 2
        # Initialize the first layer because it is different than the others.
        # TESTING GAUSS
        layers.append(SpectralNorm(nn.Sequential(*[GaussianNoise(), nn.Conv2d(3, conv_dim, 4, 2, 1)])))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        for n in range(n_layers - 1):
            # TESTING GAUSS
            layers.append(SpectralNorm(nn.Sequential(*[GaussianNoise(), nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)])))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim *= 2
            if 2**(n+2) in attn_feat:
                layers.append(Self_Attn(curr_dim, 'relu'))

        layers.append(GaussianNoise()) # TESTING
        layers.append(nn.Conv2d(curr_dim, 1, 4))
        self.output = nn.Sequential(*layers)

    # def forward(self, x):
    #     out = self.l1(x)
    #     out = self.l2(out)
    #     out = self.l3(out)
    #     out,p1 = self.attn1(out)
    #     out=self.l4(out)
    #     out,p2 = self.attn2(out)
    #     out=self.last(out)

    #     return out.squeeze(), p1, p2

    def forward(self, x):
        out = self.output(x)
        p1 = []
        p2 = []
        return out.squeeze(), p1, p2
