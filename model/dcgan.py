import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model.loss import *

class DCGAN:
	def __init__(self, 
			device = 'cuda',
			ngpu = 1,  # Number of GPUs available. Use 0 for CPU mode.
			dataloader = None,
			nc = 3,  # Number of channels in the training images. For color images this is 3
			nz = 100,   # Size of z latent vector (i.e. size of generator input)
			ngf = 64,   # Size of feature maps in generator
			ndf = 64,   # Size of feature maps in discriminator
			lr = 0.0002, # Learning rate for optimizers
			leakyrelu_alpha = 0.2,
			beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
			loss_fuction = None):
		self.device = device
		self.dataloader = dataloader
		self.loss_fuction = loss_fuction
		self.nz = nz

		self.netG = Generator(ngpu = ngpu, ngf = ngf, nc = nc, nz = self.nz).to(device)
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

	def train(self, num_epochs):
		G_losses = []
		D_losses = []
		iters = 0

		print("Starting Training Loop...")
		for epoch in range(num_epochs):
			for i, data in enumerate(self.dataloader, 0):
				

				#Discriminator training
				self.netD.zero_grad()

				real = data[0].to(self.device)
				b_size = real.size(0)
				real_label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
				output_real = self.netD(real).view(-1)


				noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
				fake = self.netG(noise)
				fake_label = torch.full((b_size,), 0, dtype=torch.float, device=self.device)
				output_fake = self. netD(fake.detach()).view(-1)
				
				D_loss = self.loss_fuction(which_model = 'D', out_fake  = output_fake, out_real = output_real)
				D_loss.backward()
				self.optimizerD.step()

				#Generator training

				self.netG.zero_grad()

				output_fake = self.netD(fake).view(-1)
				
				G_loss = self.loss_fuction(which_model = 'G', out_fake  = output_fake, out_real = None)
				G_loss.backward()
				self.optimizerG.step()
				if i % 50 == 0:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
						% (epoch, num_epochs, i, len(self.dataloader),
						D_loss.item(), G_loss.item()))
				G_losses.append(G_loss.item())
				D_losses.append(D_loss.item())

				iters += 1

	def generate_images(self, fixed_noise):
		with torch.no_grad():
			fake = self.netG(fixed_noise).detach().cpu()
		img_list = fake

		return img_list












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
		y = self.cov1_layer(input)
		y = self.cov2_layer(y)
		y = self.cov3_layer(y)
		y = self.cov4_layer(y)
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
		y = self.cov1_layer(input)
		y = self.cov2_layer(y)
		y = self.cov3_layer(y)
		y = self.cov4_layer(y)
		y = self.out_layer(y)
		return y

