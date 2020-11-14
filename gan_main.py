from __future__ import print_function
import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np




from models.loss import *
from models.dcgan import *
from data_processing.save_output import *


# Set random seed for reproducibility


working_path = '/nfs/GAN/'


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)




arg_attention = True

arg_spectralnorm = True

arg_n_critic = 2

arg_orthogonal_regularization = 1e-5

arg_loss = 'relativistic'

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr_g = 0.0002
lr_d = 0.0002

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2




# Root directory for dataset
dataroot =os.path.join(working_path, "celeba")



dataset = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
															 transforms.Resize(image_size),
															 transforms.CenterCrop(image_size),
															 transforms.ToTensor(),
															 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
													 ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)



device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if 'standard' in arg_loss:
	loss_fuction = standard_adversarial_loss

elif 'hinge' in arg_loss:
	loss_fuction = hinge_adversarial_loss

elif 'relativistic' in arg_loss:
	loss_fuction = relativistic_average_discriminater


print()


gan_model = DCGAN(device = device,
						ngpu = 1,  # Number of GPUs available. Use 0 for CPU mode.
						dataloader = dataloader,
						nc = 3,  # Number of channels in the training images. For color images this is 3
						nz = 100,   # Size of z latent vector (i.e. size of generator input)
						ngf = 64,   # Size of feature maps in generator
						ndf = 64,   # Size of feature maps in discriminator
						lr_g = 0.0002, # Learning rate for optimizers
						lr_d = 0.0002,
						leakyrelu_alpha = 0.2,
						beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
						is_attention = arg_attention,
						is_spectral_norm = arg_spectralnorm,
						n_critic = arg_n_critic,
						orthogonal_regularization_scale = arg_orthogonal_regularization,
						loss_fuction = loss_fuction)

print('Device: ', device)
print('Loss Functiion:', arg_loss)
print('Attention: ', arg_attention)
print('Spectral Norm: ', arg_spectralnorm)
print('n_critic: ', arg_n_critic)


sub_path = 'model_output/DCGAN'

import os
import glob






sub_path = sub_path + '_' + arg_loss

sub_path = sub_path + '_imagesize' + str(image_size)

sub_path = sub_path + '_batchsize' + str(batch_size)

sub_path = sub_path + '_weightsinitorthogonal'

sub_path = sub_path + '_orthogonalregularization' + str(arg_orthogonal_regularization)


if arg_attention:
	sub_path = sub_path + '_attention'

if arg_spectralnorm:
	sub_path = sub_path + '_spectral'

sub_path = sub_path + '_log/'

output_path = os.path.join(working_path, sub_path)
fixed_noise = np.random.normal(0, 1, size=(100, nz, 1, 1))

import shutil
shutil.rmtree(os.path.join(working_path, 'model_output'))


real_tmp = next(iter(dataloader))[0]


store_output_images(output_path, fake = None, real = real_tmp, prefix = "")

gan_model.train(num_epochs, fixed_noise, output_path = output_path)












