# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:18:55 2017

@author: Pawan
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy import misc

"""
 RGB to Grayscale conversion 
"""
def rgb2grayfloat(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    matrix = np.linalg.inv([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
    r1, g1, b1 = matrix[0,0], matrix[0,1], matrix[0,2]
    gray = r1 * r + g1 * g + b1 * b
    return gray.astype(float)

"""
computation of weighted Gradient Similarity Index with weights from 
Gaussian and Motion Blur convolution kernels
"""
def gsi_index(x1,x2,tt):
    ddx = np.array([[3, 0, -3],[10, 0, -10],[3, 0, -3]], dtype=float)
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
    x1 = x1.astype(float)
    x2 = x2.astype(float)
    m1 = gau_index(x1,(11,11),5.5)
    m2 = mot_index(x1)
    y1 = gsi_index(x1, x2, 250)
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
def sqms_gray(x1,x2):
    # x1 and x2 can be RGB images or grayscale images as numpy arrays
    if len(x1.shape) > 2 and x1.shape > 1:
        # more than one channel
        x1 = rgb2grayfloat(x1[:,:,0:3])
        x2 = rgb2grayfloat(x2[:,:,0:3])
    sqms = gen_sqms(x1,x2)
    print('sqms_gray = '+repr(sqms))
    return sqms

"""
SQMS metric computation for Cb abd Cr of 
original and reconstructed image
"""
def sqms_chroma(YCbCr_orig,YCbCr_reco):
    x1_cb = YCbCr_orig[:,:,1]
    x1_cr = YCbCr_orig[:,:,2]
    x2_cb = YCbCr_reco[:,:,1]
    x2_cr = YCbCr_reco[:,:,2]
    sqms_cb = gen_sqms(x1_cb,x2_cb)
    sqms_cr = gen_sqms(x1_cr,x2_cr)
    print('sqms_cb = '+repr(sqms_cb)+'  sqms_cr = '+repr(sqms_cr))
    sqms_chroma = (sqms_cb + sqms_cr)/2
    print('sqms_chroma = '+repr(sqms_chroma))
    return sqms_chroma
