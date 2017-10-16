This is the python implementation of the SQMS metric MATLAB code provided by the authors of the paper. 

The SQMS metric can be computed either for the grayscale of original and reconstructed image or from the chroma (Cb and Cr) as follows :
1) sqms_gray(orig_img,reco_img)
2) sqms_chroma(orig_img,reco_img)

Note : 

1) For sqms_chroma, avearging of sqms_Cb and sqms_Cr is used in the current code. However, the minimum function (min()) can also be used and it gives a slightly better correlation values.

2) SQMS computation makes use of gaussian window and motion blur window. While the gaussian window has been implemented in the python code itself, the motion blur window has been imported from the .mat file : 'window.mat'. The matlab file 'mb_window.m' is used to compute the 'window.mat'file.



