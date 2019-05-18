
from scipy import ndimage
import numpy as np
from pylab import *
from PIL import Image

for i in range(30):
	fileloc = '/home/oxygen39/B241212/optimized_images/'
	filename = 'mda{:d}_optimized_object.csv'.format(371+(i*2))

	angle_fn = 'mda{:d}_optimized_image_angle'.format(371+(i*2))
	abs_fn = 'mda{:d}_optimized_image_abs'.format(371+(i*2))


	ob = np.genfromtxt(fileloc+filename, delimiter=',', dtype = complex)
	ang_ob = np.angle(ob)
	#abs_ob = abs(ob)

	ang_ob_x = ndimage.sobel(ang_ob, axis=0, mode='constant')
	ang_ob_y = ndimage.sobel(ang_ob, axis=1, mode='constant')

	sob_fil = np.hypot(ang_ob_x, ang_ob_y)
	sob_fil = sob_fil[812:1240,842:1250]



	imsave('/home/oxygen39/B241212/optimized_images/sobel_corrections/'+angle_fn+'.png', sob_fil, cmap=cm.hsv)
	#imsave('/home/guru/Documents/optimized_pngs/'+abs_fn+'.png', abs_ob)



#apply gaussian or other filter
