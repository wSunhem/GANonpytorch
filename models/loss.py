import numpy as np
import torch
import torch.nn as nn

BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
Relu = nn.ReLU()

def standard_adversarial_loss(which_model, out_fake  = None, out_real = None):
	if 'D' in which_model:
		loss_fake = BCEWithLogitsLoss(out_fake, torch.zeros_like(out_fake))
		loss_real = BCEWithLogitsLoss(out_real, torch.ones_like(out_real))
		final_loss = (loss_fake + loss_real)/2
	elif 'G' in which_model:
		loss_fake = BCEWithLogitsLoss(out_fake, torch.ones_like(out_fake))
		final_loss = loss_fake
	return final_loss

def hinge_adversarial_loss(which_model, out_fake  = None, out_real = None):
	if 'D' in which_model:
		loss_real = Relu(1.0 - out_real).mean()
		loss_fake = Relu(1.0 + out_fake).mean()
		final_loss = (loss_fake + loss_real)/2
	elif 'G' in which_model:
		loss_fake = -out_fake.mean()
		final_loss = loss_fake
	return final_loss

def relativistic_average_discriminater(which_model, out_fake  = None, out_real = None):

	diff_xr = out_real - out_fake.mean(axis = 0, keepdim = True)
	diff_xf = out_fake - out_real.mean(axis = 0, keepdim = True)


	if 'D' in which_model:
		loss_fake = BCEWithLogitsLoss(diff_xf, torch.zeros_like(out_fake)) 
		loss_real = BCEWithLogitsLoss(diff_xr, torch.ones_like(out_real)) 
		final_loss = (loss_fake + loss_real)/2
	elif 'G' in which_model:
		loss_fake = BCEWithLogitsLoss(diff_xf, torch.ones_like(out_fake)) 
		loss_real = BCEWithLogitsLoss(diff_xr, torch.zeros_like(out_real)) 
		final_loss = (loss_fake + loss_real)/2
	return final_loss
