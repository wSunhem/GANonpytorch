import numpy as np
import torch
import torch.nn as nn

BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
Relu = nn.ReLU()

def standard_adversarial_loss(which_model, out_fake  = None, out_real = None):
	if 'D' in which_model:
		loss_fake = BCEWithLogitsLoss(out_fake, torch.zeros_like(out_fake), reduction = 'mean')
		loss_real = BCEWithLogitsLoss(out_real, torch.ones_like(out_fake), reduction = 'mean')
		final_loss = (loss_fake + loss_real)/2
	elif 'G' in which_model:
		loss_fake = BCEWithLogitsLoss(out_fake, torch.ones_like(out_fake), reduction = 'mean')
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

