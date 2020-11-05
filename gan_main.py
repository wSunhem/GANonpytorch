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


from model.loss import *
from model.dcgan import *
from data_processing.save_output import *


# Set random seed for reproducibility


working_path = '/nfs/GAN/'


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

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
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


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

loss_fuction = standard_adversarial_loss

gan_model = DCGAN(device = device,
						ngpu = 1,  # Number of GPUs available. Use 0 for CPU mode.
						dataloader = dataloader,
						nc = 3,  # Number of channels in the training images. For color images this is 3
						nz = 100,   # Size of z latent vector (i.e. size of generator input)
						ngf = 64,   # Size of feature maps in generator
						ndf = 64,   # Size of feature maps in discriminator
						lr = 0.0002, # Learning rate for optimizers
						leakyrelu_alpha = 0.2,
						beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
						loss_fuction = loss_fuction)

gan_model.train(num_epochs)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)


fake = gan_model.generate_images(fixed_noise)
real = real_batch = next(iter(dataloader))[0]


output_path = os.path.join(working_path,'model_output/DCGAN_log/')


store_output_images(output_path, fake, real)





