
import numpy as np
from scipy import misc
import sqms

import conversion


def main():

	im1 = '../TestImages/im1.png'
	im2 = '../TestImages/im2.png'
	#im1 = '../TestImages/im1_gray.png'
	#im2 = '../TestImages/im2_gray.png'

	x1 = misc.imread(im1)
	x2 = misc.imread(im2)


	sqms.sqms_gray(x1, x2)


	YCbCr_orig = rgb2ycbcr(x1)
	YCbCr_reco = rgb2ycbcr(x2)
	sqms.sqms_chroma(YCbCr_orig, YCbCr_reco)



def rgb2ycbcr(rgb):
	"""Convert 3D RGB image array to 3D YCbCr image array

	:param ndarray rgb: 3D RGB array
	:return: 3D YCbCr array
	:rtype: ndarray
	"""
	height, width, channels = rgb.shape
	assert channels == 3
	ycbcr = np.zeros((height, width, channels), dtype=np.uint8)

	# R, G, B = rgb
	R, G, B = np.dsplit(rgb.astype(np.int32), 3)
	Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16
	Cb = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128
	Cr = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128

	## correct overflow to headroom
	Y[Y > 235] = 235
	Cb[Cb > 240] = 240
	Cr[Cr > 240] = 240

	ycbcr = np.dstack((Y, Cb, Cr))

	ycbcr[ycbcr < 16] = 16

	return ycbcr.astype(np.uint8)


if __name__ == '__main__':
	main()