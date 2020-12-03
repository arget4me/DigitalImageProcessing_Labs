function [entropy] = imGreyEntropy(img)
%Calculates the entropy of an image
%   [entropy] = imGreyEntropy(img)
%   img - grey-scale image to calculate entropy of
%   entropy - output entropy of the image

hist = imGrey2histogram(img);

imgSize = size(img);
num_pixels = imgSize(1) * imgSize(2);

Pxi = hist./num_pixels; Pxi(Pxi==0) = [];

entropy = -sum(Pxi .* log2(Pxi));


end

