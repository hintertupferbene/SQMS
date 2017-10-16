
import numpy as np
from scipy import misc
import sqms


def main():

	im1 = 'im1.png'
	im2 = 'im2.png'
	#im1 = 'im1_gray.png'
	#im2 = 'im2_gray.png'

	x1 = misc.imread(im1)
	x2 = misc.imread(im2)

	if len(x1.shape) > 2 and x1.shape > 1:
		# more than one channel
		x1 = rgb2gray(x1)
		x2 = rgb2gray(x2)

	res = float(sqms.gen_sqms(x1.astype(float) ,x2.astype(float)))
	print( 'sqms_gray = ' +repr(res))
	return res


def rgb2gray(rgb):
	r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
	matrix = np.linalg.inv([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
	r1, g1, b1 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
	gray = r1 * r + g1 * g + b1 * b
	gray = np.array(gray)
	return gray.astype(float)


if __name__ == '__main__':
	main()