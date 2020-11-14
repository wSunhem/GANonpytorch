import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

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
  
 class Convolutional_layer(nn.Module): 
 	def __init__(self, input_channels, output_channels, filter_size, stride, padding, spectral = False, init = 'normal', regularizer=None, conv_type='Conv2d'):
 		super(Convolutional_layer, self).__init__()
 		self.conv_type = conv_type
 		self.spectral = spectral
 		self.regularizer = regularizer
 		if conv_type=='Conv2d':
 			module_tmp = nn.Conv2d(input_channels, output_channels, filter_size, stride, padding, bias=True)
 		elif conv_type=='ConvTranspose2d':
 			module_tmp = nn.ConvTranspose2d(input_channels, output_channels, filter_size, stride, padding, bias=True)

 		if init == 'normal':
 			nn.init.normal_(module_tmp.weight.data, 0.0, 0.02)
			nn.init.constant_(module_tmp.bias.data, 0)
		elif init == 'orthogonal':
			nn.init.orthogonal_(module_tmp.weight.data, 0.0, 0.02)
			nn.init.constant_(module_tmp.bias.data, 0)

		if spectral:
			self.main = spectral_norm(model_tmp)
		else:
			self.main = model_tmp



	def forward(self,x):
		return self.main(x)

