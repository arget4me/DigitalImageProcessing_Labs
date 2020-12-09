function [img_out] = convolution_operator(img_in, kernel)
% img_out = convolution_operator(img_in, kernel) convolves the input image img_in
% with the filter kernel and returns the result in img_out
% img_in, img_out, and kernel are of double precision
%       Every single pixel in the output image, img_out, is calculated from a
%	weighted sum the pixel in the input image with the same pixel
%	coordinate as the output pixel, and the neighboring pixels around it in
%	the input image. 
%       The weights are given and in the kernel matrix, and
%	the dimensions of the kernel describes the amount of neighboring pixels to
%	include in the weighted sum. The center of the kernel describe the
%	weight of the pixel with the same pixel coordinate as the output pixel.
%       The relative position around the center of the kernel describes the
%	weight for the neighboring pixel with the same relative position in the
%	input image.

if ~isa(img_in,'double') %Make sure input image is a double with values in range [0, 1]
   img_in = im2double(img_in);
end

if ~isa(kernel,'double') %Make sure kernel weights are doubles
    kernel = double(kernel);
end
convolution_kernel = rot90(rot90(kernel)); %Convolution rotates the kernel 180 degrees, correlation keeps it the same.

%initialize variables sizes
a = floor(size(convolution_kernel, 1)./2); %Num positions above/below the center of the kernel
b = floor(size(convolution_kernel, 2)./2); %Num positions right/left of the center of the kernel
kernel_center = [a; b] + [1; 1]; %By the previous logic the center is one index larger to the right and down from [a, b]. (Index of the top left kernel position is [1; 1])

img_size = size(img_in);
num_rows = img_size(1);
num_columns = img_size(2);
img_out = zeros(img_size);

%loop through every pixel in the input image and calculate the weighted sum for the
%output pixel. 
for column = 1:num_columns
    for row = 1:num_rows
        
        %Apply the kernel: multiply all values around the pixel at
        %img_in(row, column) with it's corresponding weights in the kernel.
        %   sum up all img_in(row + s, column + t) * kernel(s, t), where s is the kernel row and t is the kernel column
        for s = -a:a
            for t = -b:b
                
                %offset pixel coordinate by kernel position to find the
                %neighboring pixel coordinate
                column_neighbor = column + t;
                row_neighbor = row + s;
                
                %To index the kernel properly it needs to be padded with
                %the position of the center of the kernel since Matlab
                %matrices start at index 1.
                s_index = s + kernel_center(1);
                t_index = t + kernel_center(2); 
                
                
                %Check if a neighboring pixel exceeds the image border. If it does then Mirror the kernel around the border.
                if column_neighbor < 1
                    %Exceeds border to the left side, meaning t was negative and -t will mirror it right.
                    %   (-1 means that the first mirrored pixel will be the same as the one for the center column pixel)
                    column_neighbor = column - t - 1; 
                elseif column_neighbor > num_columns
                    %Exceeds border to the right side, meaning t was positive and -t will mirror it left.
                    %   (+1 means that the first mirrored pixel will be the same as the one for the center column pixel)
                    column_neighbor = column - t + 1;
                end
                if row_neighbor < 1
                    %Exceeds border at the top, meaning s was negative and -s will mirror it down.
                    %   (-1 means that the first mirrored pixel will be the same as the one for the center column pixel)
                    row_neighbor = row - s -1;
                elseif row_neighbor > num_rows
                    %Exceeds border at the bottom, meaning s was positive and -s will mirror it up.
                    %   (+1 means that the first mirrored pixel will be the same as the one for the center column pixel)
                    row_neighbor = row - s +1;
                end
                
                %Sum up img_in pixels multiplied by kernel weights
                img_out(row, column, :) = img_out(row, column, :) + convolution_kernel(s_index, t_index) .* img_in(row_neighbor, column_neighbor, :);
            end
        end
        
    end
end
end