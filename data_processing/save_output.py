import os, sys
from pathlib import Path
import torchvision.utils as vutils

def store_output_images(path, fake, real):
	fake_path = os.path.join(path,'fake/')
	Path(fake_path).mkdir(parents=True, exist_ok=True)

	for i in range(len(fake)):
		vutils.save_image(fake[i], fp = open(os.path.join(fake_path, str(i)+ '.jpg'), 'wb'))

	real_path = os.path.join(path,'real/')
	Path(real_path).mkdir(parents=True, exist_ok=True)

	for i in range(len(real)):
		vutils.save_image(real[i], fp = open(os.path.join(real_path, str(i) + '.jpg'), 'wb'))

	print('Output images\'ve been successfully stored: N(Generated) = %d, N(Real) = %d' % (len(fake), len(real)))