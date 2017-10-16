# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:18:55 2017

@author: Pawan
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy import misc
import pymatlab

"""
 RGB to Grayscale conversion 
"""
def rgb2grayfloat(filename):
    rgb = misc.imread(filename)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    matrix = np.linalg.inv([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
    r1, g1, b1 = matrix[0,0], matrix[0,1], matrix[0,2]
    gray = r1 * r + g1 * g + b1 * b
    gray = np.array(gray)
    gray.astype(float)
    return gray

""" 
RGB to Y, Cb abd Cr conversion #
"""
def rgb2ycbcrfloat(filename):
    rgb  = misc.imread(filename)
    height, width, channels = rgb.shape
    assert channels == 3
    ycbcr = np.zeros((height,width,channels), dtype=np.uint8)
    
    R, G, B = np.dsplit(rgb.astype(np.int32),3)
    Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16
    Cb = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128
    Cr = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128
    
    Y[Y>235] = 235
    Cb[Cb>240] = 240
    Cr[Cr>240] = 240
    
    ycbcr = np.dstack((Y, Cb, Cr))
    ycbcr[ycbcr<16] = 16
    return ycbcr.astype(np.uint8)
	   
"""
computation of weighted Gradient Similarity Index with weights from 
Gaussian and Motion Blur convolution kernels
"""
def gsi_index(x1,x2,tt):
    ddx = np.array([[3, 0, -3],[10, 0, -10],[3, 0, -3]])
    ddx = ddx/16
    ddy = np.transpose(ddx)
    xx1 = signal.convolve2d(x1,ddx,'same')
    xy1 = signal.convolve2d(x1,ddy,'same')
    Gmap1 = np.sqrt(np.square(xx1)+np.square(xy1))  
    xx2 = signal.convolve2d(x2,ddx,'same')
    xy2 = signal.convolve2d(x2,ddy,'same')
    Gmap2 = np.sqrt(np.square(xx2)+np.square(xy2))
    G1 = np.add(np.multiply(np.multiply(Gmap1,Gmap2),2),tt)
    G2 = np.add(np.square(Gmap1)+np.square(Gmap2),tt)
    yy = np.divide(G1,G2)
    return yy

"""
Gaussian Blur convolution kernel
"""
def gau_index(x1,shape,sigma):
    """
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh    
    yy = signal.convolve2d(x1,np.rot90(h),'same')    
    return yy

"""
Motion Blur convolution kernel
"""
def mot_index(x1):
    mfile = sio.loadmat('window.mat')
    h = mfile['ww']
    h = np.array(h)
    yy = signal.convolve2d(x1,h,'same')
    return yy

"""
General SQMS metric computation
"""
def gen_sqms(x1,x2):
    y1 = gsi_index(x1,x2,250)
    m1 = gau_index(x1,(11,11),5.5)
    m2 = mot_index(x1)
    y2 = np.subtract(np.ones(x1.shape),gsi_index(x1,m1,200))
    y3 = np.subtract(np.ones(x1.shape),gsi_index(x1,m2,1))
    y4 = np.divide(np.add(y2,y3),2)
    ss1 = sum(np.multiply(y1,y4))
    ss2 = sum(sum(y4))
    ss = np.real(sum(np.divide(ss1,ss2)))
    return ss

"""
SQMS metric computation for original and reconstructed image in grayscale
"""
def sqms_gray(orig_img,reco_img):
    x1 = rgb2grayfloat(orig_img)
    x2 = rgb2grayfloat(reco_img)
    sqms = float(gen_sqms(x1,x2))
    print('sqms_gray = '+repr(sqms))
    return sqms

"""
SQMS metric computation for Cb abd Cr of 
original and reconstructed image
"""
def sqms_chroma(orig_img,reco_img):
    x1 = rgb2ycbcrfloat(orig_img)
    x2 = rgb2ycbcrfloat(reco_img)
    x1_cb = x1[:,:,1]
    x1_cr = x1[:,:,2]
    x2_cb = x2[:,:,1]
    x2_cr = x2[:,:,2]
    sqms_cb = float(gen_sqms(x1_cb,x2_cb))
    sqms_cr = float(gen_sqms(x1_cr,x2_cr))
    print('sqms_cb = '+repr(sqms_cb)+'  sqms_cr = '+repr(sqms_cr))
    sqms_chroma = (sqms_cb + sqms_cr)/2
    print('sqms_chroma = '+repr(sqms_chroma))
    return sqms_chroma