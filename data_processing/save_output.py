import os, sys
from pathlib import Path
import torchvision.utils as vutils



def store_output_images(path ="", fake = None, real = None, prefix = ""):
	fake_length = 0
	real_length = 0

	if fake != None:
		fake_path = os.path.join(path, prefix + 'fake/')
		Path(fake_path).mkdir(parents=True, exist_ok=True)
		for i in range(len(fake)):
			vutils.save_image(denorm(fake[i]), fp = open(os.path.join(fake_path, str(i)+ '.jpg'), 'wb'))


		fake_length = len(fake)

	if real != None:
		real_path = os.path.join(path,prefix + 'real/')
		Path(real_path).mkdir(parents=True, exist_ok=True)

		for i in range(len(real)):
			vutils.save_image(denorm(real[i]), fp = open(os.path.join(real_path, str(i) + '.jpg'), 'wb'))
		real_length = len(real)

	print('Output images\'ve been successfully stored: N(Generated) = %d, N(Real) = %d' % (fake_length, real_length))

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)