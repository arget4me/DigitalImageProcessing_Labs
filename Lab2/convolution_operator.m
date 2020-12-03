function [img_out] = convolution_operator(img_in, kernel)
% img_out = convolution_operator(img_in, kernel) convolves the input image img_in
% with the filter kernel and returns the result in img_out
% img_in, img_out, and kernel are of double precision
if ~isa(img_in,'double')
   img_in = double(img_in);
end

if ~isa(kernel,'double')
    kernel = double(kernel);
end

%initialize variables
kernelSize = floor(size(kernel)./2); b = kernelSize(1); a = kernelSize(2);
imgSize = size(img_in); num_rows = imgSize(1); num_columns = imgSize(2);
img_out = zeros(imgSize);

%loop through every pixel in image
for column = 1:num_columns
    for row = 1:num_rows
        
        %apply kernel
        for s = -a:a
            for t = -b:b
                
                %offset pixel coordinate by kernel position
                x_index = column + t; y_index = row + s;
                s_index = s + 1 + a; t_index = t + 1 + b; %matlab matrices start at 1
                
                %check image border constraint: mirror if outside
                if x_index < 1 || x_index > num_columns
                    x_index = column - t;
                end
                if y_index < 1 || y_index > num_rows
                    y_index = row - s;
                end
                
                %Sum up img_in pixels multiplied by kernel weights
                img_out(row, column, :) = img_out(row, column, :) + kernel(s_index, t_index) .* img_in(y_index, x_index, :);
            end
        end
        
    end
end
end