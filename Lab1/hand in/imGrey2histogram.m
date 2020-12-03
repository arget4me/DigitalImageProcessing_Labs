function [outHist] = imGrey2histogram(img)
%Compute histogram for image
%   [outHist] = imGrey2histogram(img)
%   img - grey scale image to compute compute histogram of
%   outHist - output histogram

outHist = zeros(1, 256);
imgSize = size(img);

for x = 1:imgSize(2)
   for y = 1:imgSize(1)
       index = double(img(y, x)) + 1;
       outHist(1, index) = outHist(1, index) + 1;
   end 
end

%{
Uncomment this to ignore grey-values 0 and 255.
Usable to test how much the spikes in these values affect the result of
finding the images. (Histogram RMS calculation is heavily affected)
%outHist = outHist(1, 2:255);
%}
 

end