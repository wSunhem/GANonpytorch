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
from torch.nn.utils import spectral_norm

from data_processing.save_output import *

class DCGAN:
	def __init__(self, 
			device = 'cuda',
			ngpu = 1,  # Number of GPUs available. Use 0 for CPU mode.
			dataloader = None,
			nc = 3,  # Number of channels in the training images. For color images this is 3
			nz = 100,   # Size of z latent vector (i.e. size of generator input)
			ngf = 64,   # Size of feature maps in generator
			ndf = 64,   # Size of feature maps in discriminator
			lr_g = 0.0001, # Learning rate for optimizers
			lr_d = 0.0004,
			leakyrelu_alpha = 0.2,
			beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
			is_attention = False,
			is_spectral_norm = False,
			n_critic= 1,
			loss_fuction = None):


		self.device = device
		self.dataloader = dataloader
		self.loss_fuction = loss_fuction
		self.nz = nz
		self.n_critic = n_critic

		self.netG = Generator(is_attention = is_attention, is_spectral_norm = is_spectral_norm,  ngpu = ngpu, ngf = ngf, nc = nc, nz = self.nz).to(device)
		self.netD = Discriminator(is_attention = is_attention, is_spectral_norm = is_spectral_norm, ngpu = ngpu, ndf = ndf, nc = nc, leakyrelu_alpha = leakyrelu_alpha).to(device)

		if (device.type == 'cuda') and (ngpu > 1):
			print('GPU is available')
			self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
			self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

		self.netG.apply(weights_init)
		self.netD.apply(weights_init)

		self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

		print(self.netG)
		print(self.netD)
		print("GAN creation completed")

	def train(self, num_epochs, fixed_noise, output_path):
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
				if(iters % self.n_critic ==0):
					self.netG.zero_grad()

					output_fake = self.netD(fake).view(-1)
					
					G_loss = self.loss_fuction(which_model = 'G', out_fake  = output_fake, out_real = None)
					G_loss.backward()
					self.optimizerG.step()
				if i % 50 == 0:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
						% (epoch, num_epochs, i, len(self.dataloader),
						D_loss.item(), G_loss.item()))
				if iters % 100 == 0:
					fake_tmp = self.generate_images(fixed_noise)
					store_output_images(path = output_path, fake = fake_tmp, real = None, prefix = 'inter' + str(iters) + "_")



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
	def __init__(self, is_attention, is_spectral_norm, ngpu, ngf, nc, nz):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.is_attention = is_attention
		self.is_spectral_norm = is_spectral_norm

		
		conv1_module = []
		if self.is_spectral_norm:
			conv1_module.append(spectral_norm(nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=True)))
		else:
			conv1_module.append(nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=True))
		conv1_module.append(nn.BatchNorm2d(ngf * 8))
		conv1_module.append(nn.ReLU(True))

		conv2_module = []
		if self.is_spectral_norm:
			conv2_module.append(spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True)))
		else:
			conv2_module.append(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True))
		conv2_module.append(nn.BatchNorm2d(ngf * 4))
		conv2_module.append(nn.ReLU(True))

		conv3_module = []
		if self.is_spectral_norm:
			conv3_module.append(spectral_norm(nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=True)))
		else:
			conv3_module.append(nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=True))
		conv3_module.append(nn.BatchNorm2d(ngf * 2))
		conv3_module.append(nn.ReLU(True))

		conv4_module = []
		if self.is_spectral_norm:
			conv4_module.append(spectral_norm(nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=True)))
		else:
			conv4_module.append(nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=True))
		conv4_module.append(nn.BatchNorm2d(ngf))
		conv4_module.append(nn.ReLU(True))

		self.conv1_layer = nn.Sequential(*conv1_module) 
		self.conv2_layer =  nn.Sequential(*conv2_module)
		self.conv3_layer =  nn.Sequential(*conv3_module)
		self.conv4_layer =  nn.Sequential(*conv4_module)


		self.out_layer =  nn.Sequential(nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=True), nn.Tanh())

		if(self.is_attention):
			self.attn1 = Self_attn(ngf * 2)
			self.attn2 = Self_attn(ngf)

	def forward(self, input):
		y = self.conv1_layer(input)
		y = self.conv2_layer(y)
		y = self.conv3_layer(y)

		if(self.is_attention):
			y, a = self.attn1(y)

		y = self.conv4_layer(y)

		if(self.is_attention):
			y, a = self.attn2(y)
		
		y = self.out_layer(y)
		return y

class Discriminator(nn.Module):
	def __init__(self, is_attention, is_spectral_norm, ngpu, ndf, nc, leakyrelu_alpha):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.is_attention = is_attention
		self.is_spectral_norm = is_spectral_norm

		conv1_module = []
		if self.is_spectral_norm:
			conv1_module.append(spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True)))
		else:
			conv1_module.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True))
		conv1_module.append(nn.LeakyReLU(leakyrelu_alpha, inplace=True))

		conv2_module = []
		if self.is_spectral_norm:
			conv2_module.append(spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)))
		else:
			conv2_module.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True))
		conv2_module.append(nn.BatchNorm2d(ndf * 2))
		conv2_module.append(nn.LeakyReLU(leakyrelu_alpha, inplace=True))

		conv3_module = []
		if self.is_spectral_norm:
			conv3_module.append(spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)))
		else:
			conv3_module.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True))
		conv3_module.append(nn.BatchNorm2d(ndf * 4))
		conv3_module.append(nn.LeakyReLU(leakyrelu_alpha, inplace=True))

		conv4_module = []
		if self.is_spectral_norm:
			conv4_module.append(spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)))
		else:
			conv4_module.append(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True))
		conv4_module.append(nn.BatchNorm2d(ndf * 8))
		conv4_module.append(nn.LeakyReLU(leakyrelu_alpha, inplace=True))




		self.conv1_layer = nn.Sequential(*conv1_module) 
		self.conv2_layer =  nn.Sequential(*conv2_module)
		self.conv3_layer =  nn.Sequential(*conv3_module)
		self.conv4_layer =  nn.Sequential(*conv4_module)

		self.out_layer = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)  #nn.Sigmoid()
		if(self.is_attention):
			self.attn1 = Self_attn(ndf * 4)
			self.attn2 = Self_attn(ndf * 8)


	def forward(self, input):
		y = self.conv1_layer(input)
		y = self.conv2_layer(y)
		y = self.conv3_layer(y)

		if(self.is_attention):
			y, a = self.attn1(y)

		y = self.conv4_layer(y)

		if(self.is_attention):
			y, a = self.attn2(y)
		
		y = self.out_layer(y)
		return y


class Self_attn(nn.Module):

	def __init__(self,in_dim):
		super(Self_attn,self).__init__()
		self.chanel_in = in_dim
		self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) 
	def forward(self,x):

		m_batchsize,C,width ,height = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) 
		energy =  torch.bmm(proj_query,proj_key) 
		attention = self.softmax(energy) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) 

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,width,height)
		
		out = self.gamma*out + x
		return out, attention
  
