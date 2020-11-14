import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models.loss import *
from torch.nn.utils import spectral_norm

from data_processing.save_output import *

from opt import *

class DCGAN:
	def __init__(self, 
			device = 'cuda', 
			ngpu = 1, 
			dataloader = None, 
			nc = 3, 
			nz = 100, 
			ngf = 64, 
			ndf = 64, 
			lr_g = 0.0002, 
			lr_d = 0.0002, 
			leakyrelu_alpha = 0.2, 
			beta1 = 0.5, 
			is_attention = False, 
			is_spectral_norm = False, 
			n_critic = 1, 
			orthogonal_regularization_scale = None, 
			loss_fuction = None):

		self.device = device
		self.dataloader = dataloader
		self.loss_fuction = loss_fuction
		self.nz = nz
		self.n_critic = n_critic
		self.orthogonal_regularization_scale = orthogonal_regularization_scale

		self.netG = Generator(is_attention = is_attention, is_spectral_norm = is_spectral_norm,  ngpu = ngpu, ngf = ngf, nc = nc, nz = self.nz).to(device)
		self.netD = Discriminator(is_attention = is_attention, is_spectral_norm = is_spectral_norm, ngpu = ngpu, ndf = ndf, nc = nc, leakyrelu_alpha = leakyrelu_alpha).to(device)

		if (device.type == 'cuda') and (ngpu > 1):
			print('GPU is available')
			self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
			self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

		self.netG.apply(weights_init_orthogonal)
		self.netD.apply(weights_init_orthogonal)

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


				noise = torch.normal(0,1,  size = (b_size, self.nz, 1, 1), device=self.device)
				fake = self.netG(noise)
				fake_label = torch.full((b_size,), 0, dtype=torch.float, device=self.device)
				output_fake = self. netD(fake.detach()).view(-1)


				
				D_loss = self.loss_fuction(which_model = 'D', out_fake  = output_fake, out_real = output_real) 

				if self.orthogonal_regularization_scale != None:
					D_loss = D_loss + orthogonal_regularization(self.netD, self.device, self.orthogonal_regularization_scale)

				D_loss.backward()
				self.optimizerD.step()



				#Generator training
				if(iters % self.n_critic ==0):
					self.netG.zero_grad()
					output_fake = self.netD(fake).view(-1)
					if 'relativistic' in self.loss_fuction.__name__ :
						output_real = self.netD(real).view(-1)
					else:
						output_real = None

					G_loss = self.loss_fuction(which_model = 'G', out_fake  = output_fake, out_real = output_real)
					if self.orthogonal_regularization_scale != None:
						G_loss = G_loss + orthogonal_regularization(self.netG, self.device, self.orthogonal_regularization_scale)
					G_loss.backward()
					self.optimizerG.step()


				if i % 50 == 0:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
						% (epoch, num_epochs, i, len(self.dataloader),
						D_loss.item(), G_loss.item()))




				G_losses.append(G_loss.item())
				D_losses.append(D_loss.item())

				iters += 1

			fake_tmp = self.generate_images(torch.from_numpy(fixed_noise).float().to(self.device))
			store_output_images(path = output_path, fake = fake_tmp, real = None, prefix = 'epoch' + str(epoch) + "_")

	def generate_images(self, fixed_noise):
		with torch.no_grad():
			fake = self.netG(fixed_noise).detach().cpu()
		img_list = fake

		return img_list

 def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.orthogonal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


def orthogonal_regularization(model, device, beta=1e-4):

	loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
	
	for name, param in model.named_parameters():

		if 'weight' in name and ('conv' in name)  and param.requires_grad and len(param.shape)==4:

			N, C, H, W = param.shape

			weight = param.view(N * C, H, W)

			
			weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) 

			ones = torch.ones(N * C, H, H, dtype=torch.float32) 
			
			diag = torch.eye(H, dtype=torch.float32) 
			
			loss_orth += ((weight_squared * (ones - diag).to(device)) ** 2).sum()
			
	return loss_orth * beta



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


		conv5_module = []


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

		# if(self.is_attention):
		# 	y, a = self.attn1(y)

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

		# if(self.is_attention):
		# 	y, a = self.attn1(y)

		y = self.conv4_layer(y)

		if(self.is_attention):
			y, a = self.attn2(y)
		
		y = self.out_layer(y)
		return y



