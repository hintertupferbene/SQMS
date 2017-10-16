function ss = sqms_index(x1,x2)

if size(x1,3)==3; x1=rgb2gray(x1); end;
if size(x2,3)==3; x2=rgb2gray(x2); end;
x1=double(x1);
x2=double(x2);
aa = 001;
m1 = gau_index(x1,011,5.5);
m2 = mot_index(x1,009,001);
y1 = gsi_index(x1,x2,250);
y2 = 1-gsi_index(x1,m1,200);
y3 = 1-gsi_index(x1,m2,001);
y4 = (y2+aa*y3)/(1+aa);
ss = real(sum(sum((y1.*y4)/sum(sum(y4)))));
% ss = real(sum(sum(y1.*y4)))/sum(sum((y4)));

%===================================
function yy = gsi_index(img1,img2,tt)
ddx = [3 0 -3;10 0 -10;3 0 -3]/16;
ddy = ddx';
xx1 = conv2(img1, ddx, 'same');
xy1 = conv2(img1, ddy, 'same');
GMap1 = sqrt(xx1.^2 + xy1.^2);
xx2 = conv2(img2, ddx, 'same');
xy2 = conv2(img2, ddy, 'same');
GMap2 = sqrt(xx2.^2 + xy2.^2);
G1 = 2*GMap1.*GMap2;
G2 = GMap1.^2 + GMap2.^2;
yy = (G1 + tt) ./(G2 + tt);
%===================================
function yy = gau_index(xx, ll, vv)
ww = fspecial('gaussian', ll, vv);
yy = filter2(ww, xx, 'same');
function yy = mot_index(xx, ll, vv)
ww = fspecial('motion', ll, vv);
yy = filter2(ww, xx, 'same');