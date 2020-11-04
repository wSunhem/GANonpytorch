import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class DCGAN:
    def __init__(self, 
            device = 'cuda',
            ngpu = 1,  # Number of GPUs available. Use 0 for CPU mode.
            dataloader = None,
            nc = 3,  # Number of channels in the training images. For color images this is 3
            nz = 100,   # Size of z latent vector (i.e. size of generator input)
            ngf = 64,   # Size of feature maps in generator
            ndf = 64,   # Size of feature maps in discriminator
            num_epochs = 5, # Number of training epochs
            lr = 0.0002, # Learning rate for optimizers
            leakyrelu_alpha = 0.2,
            beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
            loss = None):

        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.loss = loss


        self.netG = Generator(ngpu = ngpu, ngf = ngf, nc = nc, nz = nz).to(device)
        self.netD = Discriminator(ngpu = ngpu, ndf = ndf, nc = nc, leakyrelu_alpha = leakyrelu_alpha).to(device)

        if (device.type == 'cuda') and (ngpu > 1):
            print('GPU is available')
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))


        print(self.netG)
        print(self.netD)
        print("GAN creation completed")

        



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nc, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.cov1_layer = nn.Sequential(nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True)) 
        self.cov2_layer =  nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.cov3_layer =  nn.Sequential(nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))
        self.cov4_layer =  nn.Sequential(nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True))
        self.out_layer =  nn.Sequential(nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        y = self.cov1_layer1(input)
        y = self.cov1_layer2(y)
        y = self.cov1_layer3(y)
        y = self.cov1_layer4(y)
        y = self.out_layer(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, leakyrelu_alpha):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.cov1_layer = nn.Sequential( nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  nn.LeakyReLU(leakyrelu_alpha, inplace=True))
        self.cov2_layer = nn.Sequential( nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(leakyrelu_alpha, inplace=True))
        self.cov3_layer = nn.Sequential( nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(leakyrelu_alpha, inplace=True))
        self.cov4_layer = nn.Sequential( nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(leakyrelu_alpha, inplace=True))
        self.out_layer = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)  #nn.Sigmoid()

    def forward(self, input):
        y = self.cov1_layer1(input)
        y = self.cov1_layer2(y)
        y = self.cov1_layer3(y)
        y = self.cov1_layer4(y)
        y = self.out_layer(y)
        return y

