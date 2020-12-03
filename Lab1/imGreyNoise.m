function [imgOut] = imGreyNoise(img, mu, sigma)
% Add noise to a grayscale image
%   [imgOut] = imGreyNoise(img)
%	img - image to compute grey level statistics on
%   mu - Mean of the normal distribution
%   sigma - Standard deviation of the normal distribution
%   imgOut - output image noise added




imgSize = size(img);
imgOut = uint8(zeros(imgSize));
noise_mem = zeros(imgSize(1) * imgSize(2), 1);
for column = 1:imgSize(2)
    for row = 1:imgSize(1)
        noise = min(max(normrnd(mu,sigma),-sigma),sigma);
        noise_mem(row + column * imgSize(2), 1) = noise;
        imgOut(row, column) = (img(row, column) + floor(noise * 255.0/2));
    end
end

noise_mean = mean(noise_mem)
noise_std = std(noise_mem)
noise_min = min(noise_mem)
noise_max = max(noise_mem)


end
