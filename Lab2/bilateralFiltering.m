function [img_out] = bilateralFiltering(img_in, sigma_range, sigma_domain, kernel_size)
%% Applies bilateral filtering to input image with specified sigma values for
% the range and domain distributions functions. This functionis intented for gray-scale images but if the input image has rgb-color
% channels then bilateral filtering is done for r-, g- and b-channels indepentendly of
% the others.
% [img_out] = bilateralFiltering(img_in, sigma_range, sigma_domain) 
%	img_in - input image
%   sigma_range - sigma value for the range distribution (range refers to spacial distance in pixel)
%   sigma_domain - sigma value for the domain distribution (domain refers to intensity-levels of pixels)
%   kernel_size - an odd number describing the size of the kernel. If the kernel_size = n then the dimensions of the kernel is (n x n)
%   img_out - output image

%% Extra details on bilateral filtering
%{
The kernel changes values with each reference pixel, therefor the
convolution_operator can't be used since it applies one single kernel to
the entire image. Instead the kernel is calculted by combinding two gaussins, one gaussian for
the range/distance between neighboring pixels and another gaussian for the value/intesity difference
between neighboring pixels. The mean values for the two gaussians are
the current reference pixel's position and intensity value, but since it's the relative 
difference that is of relevance the gaussians can be centered around zero.
    With small difference between intesities bilateral filtering works like a
gaussian blur. But where the difference in intensity are large, bilatteral filtering ignore neighboring pixels.
The result is that edges are preserved and other parts smoothen. By reapplying bilateral filtering to
the same image over and over leads to more flattened regions in the image,
similar to a cartoony look.
%}

%% Implementation

%Initialize variables and sizes
num_rows = size(img_in, 1);
num_columns = size(img_in, 2);
img_boundaries = [num_rows, num_columns];
img_out = zeros(size(img_in));

%lambda functions for gaussians
gaussian = @(x, mu, sigma_squarred) 1 / sqrt(2*pi.*sigma_squarred) .* exp(- (x - mu).^2./(2 .* sigma_squarred));

%x neighboring pixel intensity value and mu is the reference pixel intensity value
domain_gaussian = @(x, mu) gaussian(x, mu, sigma_domain^2); 

%The range needs to be specified in relation to the reference pixel (the center pixel in the kernel).
%This is achived by setting mu equal to zero for the range-gaussian, and
%the difference in range between the pixels is considered to be the
% euclidean distance from the reference pixel.
%x is the euclidean distance from the reference pixel
range_gaussian = @(x) gaussian(x, 0, sigma_range^2);
kernel_center = [round(kernel_size/2), round(kernel_size/2)];
kernel_center_offset = kernel_center + [-1, -1]; %offset = center - first_index. (the first index is [1, 1] since matlab matrices start with index 1)

%precalculate the range gaussian, it is the same for every reference pixel
%since the relative distance used.
range_gaussian_kernel = zeros(kernel_size, kernel_size);
for kernel_column = 1:kernel_size
            for kernel_row = 1:kernel_size
                 %Calculate relative position of neighboring pixel.
                kernel_position = [kernel_row, kernel_column];
                neighbor_relative_position = kernel_position - kernel_center_offset;
                range_gaussian_kernel(kernel_position(1), kernel_position(2)) = range_gaussian(norm(neighbor_relative_position));
            end
end
%range_contribution = range_gaussian(norm(neighbor_relative_position)); %@NOTE this is almost constant and can be precomputed, only border that might be special



fprintf("Applying bilateral filtering to input image with range-sigma = %f and domain-sigma = %f\n", sigma_range, sigma_domain);
for column = 1:num_columns
    %Display progress, every 1/10 iterations
    if mod(column, floor(num_columns/10)) == 0
        fprintf(". ");
    end
    for row =  1:num_rows
        
        reference_position = [row, column];
        reference_pixel_intensity = img_in(reference_position(1), reference_position(2), :);
        
        Wp = 0; %reset normalization factor
        
        %Modify and apply kernel relative to reference pixel.
        for kernel_column = 1:kernel_size
            for kernel_row = 1:kernel_size
                
                %Calculate relative and absolute position of neighboring pixel.
                kernel_position = [kernel_row, kernel_column];
                neighbor_relative_position = kernel_position - kernel_center_offset;
                
                neighbor_position = reference_position + neighbor_relative_position;
                %Check that neighbor_position is within image boundaries, if not mirror at the border.
                for i = 1:2 
                    if neighbor_position(i) < 1 || neighbor_position(i) > img_boundaries(i)
                        neighbor_position(i) = neighbor_position(i) - neighbor_relative_position(i) * 2;
                    end
                end
                
                neighbor_intensity = img_in(neighbor_position(1), neighbor_position(2), :);
                
                %calculate the two gaussian contribution
                domain_contribution = domain_gaussian(neighbor_intensity, reference_pixel_intensity);
                range_contribution = range_gaussian_kernel(kernel_position(1), kernel_position(2));
                
                %calculate the contribution form the neighbor by combinind
                %the gaussian output and multiply it with the intesity
                %value of the neighbor.
                neighbor_contribution =  range_contribution .* domain_contribution .* neighbor_intensity;
                
                %sum up and store the contributions in the image
                img_out(reference_position(1), reference_position(2), :) = img_out(reference_position(1), reference_position(2), :) + neighbor_contribution;
                
                %sum up total domain_contribution contribution to
                %normalization factor Wp.
                Wp = Wp + domain_contribution.*range_contribution;
            end
        end
        
        %last step is to normalize the sum based on total domain
        %contribution. (domain ratio contribution. and not absolute value
        %contribution)
        img_out(reference_position(1), reference_position(2), :) = img_out(reference_position(1), reference_position(2), :) ./ Wp;
    
    end
end
fprintf("done!\n\n");

end

