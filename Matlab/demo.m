
%clear;
%clc;

%im1 = imread('../TestImages/im1.png');
%im2 = imread('../TestImages/im2.png');
im1 = imread('../TestImages/im1_gray.png');
im2 = imread('../TestImages/im2_gray.png');

sqms = sqms_index(im1,im2)

