%% Lab Assignements #2
%% 1 Setup
clear all; close all; clc;

%Add a path to folder with all images. Test that all images can be loaded correctly.
addpath('extra_test_images')
tif_files = dir("extra_test_images/*.tif");
png_files = dir("extra_test_images/*.png");
image_files = [tif_files;png_files]

num_rows = floor(sqrt(length(image_files)));
num_columns = ceil(sqrt(length(image_files)));
for i = 1:length(image_files)
    subplot(num_rows, num_columns, i);
    I = imread(image_files(i).name);
    imshow(I); title(image_files(i).name, 'Interpreter', 'none');
end

%% 1  Linear Spatial Filtering
%%    1.1 Convolution Operator
clear all; close all; clc;

%Find all .tif and .png images in image folder.
tif_files = dir("extra_test_images/*.tif");
png_files = dir("extra_test_images/*.png");
image_files = [tif_files;png_files];

%Subplot dimensions: 
%   One row per image (assuming a maximum of 5 images in the folder).
%   Three columns to display:
%       1)convolution_operator result 
%       2)imfilter result 
%       3)image subtraction between the previous two
num_rows = length(image_files); num_columns = 3;

%The kernel demonstrated will be a gaussian kernel
%Using fspecial('gaussian', hsize, sigma) with hsize = 3 and sigma = 0.3849 gives a similar kernel to the next line
%   gauss_mask = fspecial('gaussian', 3, 0.849);
gauss_mask = [0.0625 0.125 0.0625; 0.125 0.25 0.125; 0.0625 0.125, 0.0625];

errors = zeros(1, length(image_files)); %Initialize variable to store the max errors of image subtraction for every image.

for i = 1:length(image_files)
    I = im2double(imread(image_files(i).name)); %Read image i (from list of images in folder)
    
%1) Display convolution_operator result, Border strategy is to use mirror the pixels around the border 
    img_out = convolution_operator(I, gauss_mask); 
    subplot(num_rows, num_columns, num_columns * (i - 1) + 1);
    imshow(img_out, []); title('Convolution Operator');
    
    
 %2) Display imfilter result
    img_filter = imfilter(I, gauss_mask, 'conv', 'symmetric'); %mirror at border
    subplot(num_rows, num_columns, num_columns * (i - 1) + 2);
    imshow(img_filter, []); title('Imfilter');
    
    
%3) Display image subtraction between convolution_operator result and imfilter result
    image_subtraction = img_filter - img_out;
    
    %Save max errors between the convolution_operator and imfilter for every image.
    errors(i) = max(abs(image_subtraction), [], 'all');
    
    %Display the difference between two rgb images in grayscal for better visualisation
    if(length(size(image_subtraction)) == 3)
        image_subtraction = rgb2gray(image_subtraction); 
    end
    
    %Rescale the image subtraction to be in range [0, 1] for visualisation
    image_subtraction = image_subtraction - min(image_subtraction,[],'all'); 
    image_subtraction = image_subtraction ./ max(image_subtraction,[],'all');
    
    %display image subtraction
    subplot(num_rows, num_columns, num_columns * (i - 1) + 3);
    imshow(image_subtraction, []); title('Subtract: Convolution & Imfilter');
end

%Display max errors
format long; format compact;
fprintf("Max error between convolution_operator and imfilter for each image (column i refers to image i):\n");
errors
format;

 %%   1.2 Smoothing Filters
clear all; close all; clc;

I = im2double(imread("clown.tif"));

figure('name', 'Original');
imshow(I);title('Original', 'Interpreter', 'none');

figure('name', 'Convolution operator: smoothing filter');

%The mean filters have the same weights in the entire kernel, and the sum of the kernel adds up to 1.
%The kernel can be constructed as:
%   kernel = ones(kernel_size) ./ (kernel_size^2). Or using Matlab function fspecial('average', kernel_size)
subplot(2, 3, 1);
img_mean_3 = convolution_operator(I, fspecial('average', 3));
imshow(img_mean_3, []); title('Mean 3x3', 'Interpreter', 'none');

subplot(2, 3, 2);
img_mean_5 = convolution_operator(I, fspecial('average', 5));
imshow(img_mean_5, []); title('Mean 5x5', 'Interpreter', 'none');

subplot(2, 3, 3);
img_mean_9 = convolution_operator(I, fspecial('average', 9));
imshow(img_mean_9, []); title('Mean 9x9', 'Interpreter', 'none');


%Using the same sigma as in the previous example. A larger kernelsize than 5 needs
%a larger sigma to give any significant difference since 2*sigma is
%considered to account for 95% of distribution.
%A kernel of size 5x5 has at the border 2 steps from the center, if sigma =
%0.849 then 2 steps > 2*0.849 step. Steps beyond 2 will be very small and give little difference.
sigma = 0.849;

%The gaussian kernel is defined K*exp(- (s^2 + t^2) / (2*sigma^2)) where s
%and t are the horizontal and vertical offsets from the center of the
%kernel. fspecial('gaussian', hsize, sigma) is used to get the kernels but
%it the following code achive the same resulting kernel as fspecial('gaussian', 3, 0.849):
%{
    kernel_size = 3;
    sigma = 0.849;
    kernel = ones(kernel_size, kernel_size);
    K = 1;
    for s = -floor(kernel_size/2):floor(kernel_size/2)
        for t = -floor(kernel_size/2):floor(kernel_size/2)
            kernel(round(kernel_size/2) + s, round(kernel_size/2) + t) = K * exp(- (s^2 + t^2) / (2*sigma^2));
        end
    end
    kernel = kernel ./ sum(kernel, 'all')

    difference_fspecial = kernel - fspecial('gaussian', 3, 0.849)
%}

subplot(2, 3, 4);
img_gauss_3 = convolution_operator(I, fspecial('gaussian', 3, sigma));
imshow(img_gauss_3, []); title('Gaussian 3x3', 'Interpreter', 'none');

subplot(2, 3, 5);
img_gauss_5 = convolution_operator(I, fspecial('gaussian', 5, sigma));
imshow(img_gauss_5, []); title('Gaussian 5x5', 'Interpreter', 'none');


%Displaying a scaled up version of the difference between 3x3 and 5x5
%gaussian kernel with the same sigma.
image_subtraction = img_gauss_3 - img_gauss_5;

%Display the max difference since a visual comparison between 3x3 and 5x5
%gaussian kernel isn't noticable without image subtraction.
max_error_3_vs_5 = max(abs(image_subtraction), [], 'all') 

image_subtraction = image_subtraction - min(image_subtraction, [], 'all');
image_subtraction = image_subtraction ./ max(image_subtraction, [], 'all');
subplot(2, 3, 6);
imshow(image_subtraction, []); title('Subtract: 3x3 & 5x5', 'Interpreter', 'none');



%%    1.3 Edge Filters
clear all; close all; clc;

I = im2double(imread("clown.tif"));
subplot(2, 3, 1);imshow(I, []); title('Original');

%The sobel operators give result of the difference between two sides of the
%center. It can be described as the partial derivative in the vertical or
%horizontal direction, but with extra emhasis on the center of the sides
%for better smoothing. In constant areas the partial derivative sum up to zero.
%The side where the negative terms go decide the direction/side of the edges detected.

vertical_sobel = [
    1, 2, 1;
    0, 0, 0;
    -1, -2, -1;
 ]

horizontal_sobel = [
    -1, 0, 1;
    -2, 0, 2;
    -1, 0, 1;
 ]

kernel = vertical_sobel
img_out_vertical = convolution_operator(I, kernel);
subplot(2, 3, 2); imshow(img_out_vertical, []); title('3x3 - Vertical Sobel |G_x|');

kernel = horizontal_sobel
img_out_horizontal = convolution_operator(I, kernel);
subplot(2, 3, 3); imshow(img_out_horizontal, []); title('3x3 - Horizontal Sobel |G_y|');

%The magnitude is approximated as |G| = |Gx| + |Gy|. An approximation of the expression |G| = sqrt(Gx^2 + Gy^2).
img_out = abs(img_out_vertical) + abs(img_out_horizontal);
subplot(2, 3, 4);imshow(img_out, []); title('Approximate magnitude |G| = |G_x| + |G_y|');

%The laplacian is similar to the sobel operator but can be though of as the
%2nd order derivatives

laplacian = [
    0, -1, 0;
    -1, 4, -1;
    0, -1, 0;
];%Combined horizontal and vertical laplacian

kernel = laplacian
img_out = convolution_operator(I, kernel);
subplot(2, 3, 5);imshow(img_out, []); title('3x3 - Laplacian ');


%By increasing the center of the laplacian kernel a sharpening filter is achived.
laplacian_sharpening = laplacian; laplacian_sharpening(2, 2) = laplacian_sharpening(2, 2) + 1;
%{
[
    0, -1, 0;
    -1, 5, -1;
    0, -1, 0;
]
%}

kernel = laplacian_sharpening
img_out = convolution_operator(I, kernel);
img_out = min(img_out, 1); img_out = max(img_out, 0); %set values above 1 to 1 and values below 0 to 0 in the image.
subplot(2, 3, 6);imshow(img_out, []); title('3x3 - Laplacian Sharpening');

%% 2 Bilateral Filtering
clear all; close all; clc;

sigma_range = 6; %[2,  6, 18]; 
%@Note: if range_sigma is larger than the kernel size it will have little
%affect due to the normalization factor, Wp, rescaling the values used. In
%other words the weights in the kernel is rescaled to sum up to 1. If
%range_sigma is larger then the kernel size then the assumption is that in 
%just one standardeviation the kernel_size is exceeded and will have little affect on the result. 
%   What will tend to happen with a much larger range_sigma then the kernel_size is 
%that only a small part around the mean is used and results in a filter similar to the 
%mean filter instead of a gaussian (For range only. sigma_domain, the standard deviation for the gaussian 
%refering to intensity, is limited by the range of intensities [0, 1])

sigma_domain = 0.1; %[0.1, 0.25, 10];
kernel_size = 9;

subplot(1, 3, 1); boat = im2double(imread("boat.tif")); imshow(boat); title("Boat Original");
subplot(1, 3, 2); clown = im2double(imread("clown.tif")); imshow(clown); title("Clown Original");
subplot(1, 3, 3); mandrill = im2double(imread("mandrill.tif")); imshow(mandrill); title("Mandrill Original");

boats = figure; clowns = figure; mandrills = figure;
%Loop thorugh all possible combinations of sigma_range and sigma_domain
for r = 1:length(sigma_range)
    for d = 1:length(sigma_domain)
        
        figure(boats); subplot(length(sigma_range), length(sigma_domain), d + (r-1) * length(sigma_domain));
        %apply bilateral filtering to boat image
        filtered_boat = bilateralFiltering(boat, sigma_range(r), sigma_domain(d), kernel_size); imshow(filtered_boat);
        title(['\sigma_{s} = ', num2str(sigma_range(r)), ' | \sigma_{r} = ', num2str(sigma_domain(d))]);

        figure(clowns); subplot(length(sigma_range), length(sigma_domain), d + (r-1) * length(sigma_domain));
        %apply bilateral filtering to clown image
        filtered_clown = bilateralFiltering(clown, sigma_range(r), sigma_domain(d), kernel_size); imshow(filtered_clown);
        title(['\sigma_{s} = ', num2str(sigma_range(r)), ' | \sigma_{r} = ', num2str(sigma_domain(d))]);
        
        figure(mandrills); subplot(length(sigma_range), length(sigma_domain), d + (r-1) * length(sigma_domain));
        %apply bilateral filtering to mandrill image
        filtered_mandrill = bilateralFiltering(mandrill, sigma_range(r), sigma_domain(d), kernel_size); imshow(filtered_mandrill);
        title(['\sigma_{s} = ', num2str(sigma_range(r)), ' | \sigma_{r} = ', num2str(sigma_domain(d))]);
    end
end

%% 3 Fourier Transform
%{
    @Note:
    I feel that 3.1 and 3.2 are very closely related. I have implemented
    them in the same section and present their result at the same
    time.
%}


%  3.1 Phase and Magnitude -----------------------------------------------------------------------------------------------------------

clear all; close all; clc;

mandrill = im2double(imread("mandrill.tif")); mandrill = rgb2gray(mandrill);

mandrill_fft = fftshift(fft2(mandrill));
mandrill_magnitude_matlab = abs(mandrill_fft);
mandrill_phase_matlab = angle(mandrill_fft);


%  3.2 Phase vs Magnitude -------------------------------------------------------------------------------------------------------------
%Describe formulas:

%generate a random complex number
z1 = (2 * rand() - 1) + (2 * rand() * i - i);

%magnitude is computed with Matlab function abs(complex_number)
%where abs(complex_number) = sqrt(Re(complex_number)^2 + Im(complex_number)^2)
magnitude = @(complex_number) sqrt(real(complex_number).^2 + imag(complex_number).^2)
r = magnitude(z1);
diff_abs = r - abs(z1); %Check to verify formula. difference should be 0

%phase angleis computed with Matlab function angle(complex_number)
%where angle(complex_number) = atan(Im(complex_number) / Re(complex_number))
phase = @(complex_number) atan2(imag(complex_number), real(complex_number))
theta = phase(z1);
diff_angle = theta - angle(z1);%Check to verify formula. difference should be 0


%eulers formula exp(-j * phi) = cos(phi) + i*sin(phi)
%using magnitude and phase
%it can complex number can be described as: complex_number = magnitude * (cos(phase) + i * sin(phase))
complex = @(magnitude, phase) magnitude .* (cos(phase) + i .*sin(phase))
z2 = complex(r, theta);
diff_complex = magnitude(z2 - z1); %Check to verify formula. difference should be 0

fprintf("Complex number z = %f + %fi\n", real(z1), imag(z1));
fprintf("r = magnitude(z1) = %f. Difference between r and MATLAB function abs(z1) = %f\n", r, diff_abs);
fprintf("theta = phase(z1) = %f. Difference between theta and MATLAB function angle(z1) = %f\n", theta, diff_angle);
fprintf("z2 = complex(r, theta) = %f + %fi. Difference between z2 and original z1 = %f\n", real(z2), imag(z2), diff_complex);


%Investigate importance of phase vs magnitude:

% calculate magnitude and phase for "mandrill.tif" with formulas
mandrill_magnitude = magnitude(mandrill_fft);
mandrill_phase = phase(mandrill_fft);

%load clown image and calculate phase and magnitude 
clown = im2double(imread("clown.tif"));
clown_fft = fftshift(fft2(clown));
clown_magnitude = magnitude(clown_fft);
clown_phase = phase(clown_fft);


%Switch phase and magnitude between "clown.tif" and "mandril.tif" and
%perform inverse fourier transform. There are two possible combinations (besides the two original)
replaced_phase = complex(mandrill_magnitude, clown_phase);
replaced_phase_result = real(ifft2(fftshift(replaced_phase)));

replaced_magnitude = complex(clown_magnitude, mandrill_phase);
replaced_magnitude_result = real(ifft2(fftshift(replaced_magnitude)));


%Recalculate the magnitude and phase for the new images. They appear to be
%the same as the combination used.
replaced_phase_fft = fftshift(fft2(replaced_phase_result));
replaced_magnitude_fft = fftshift(fft2(replaced_magnitude_result));

replaced_phase_fft_magnitude = magnitude(replaced_phase_fft);
replaced_phase_fft_phase = phase(replaced_phase_fft);
replaced_magnitude_fft_magnitude = magnitude(replaced_magnitude_fft);
replaced_magnitude_fft_phase = phase(replaced_magnitude_fft);


%3.2 and 3.2 Resulting plots ----------------------------------------------------------------------------------------------------------
figure("name", "Mandrill & Clown spatial domain");
subplot(2, 2, 1); imagesc(mandrill); colormap(gray); title("Mandrill");
subplot(2, 2, 2); imagesc(clown); colormap(gray); title("Clown");
subplot(2,2,3); imagesc(replaced_phase_result); colormap(gray); title("Mandrill magnitude & Clown phase");
subplot(2,2,4); imagesc(replaced_magnitude_result); colormap(gray); title("Clown magnitude & Mandrill phase");


figure("name", "Mandrill & Clown frequency domain");
subplot(2, 3, 1); imagesc(log(1 + abs(mandrill_magnitude_matlab))); colormap(gray); title("Mandrill Magnitude - MATLAB");
subplot(2, 3, 4); imagesc(mandrill_phase_matlab); colormap(gray); title("Mandrill Phase - MATLAB");

subplot(2, 3, 2); imagesc(log(1 + mandrill_magnitude)); colormap(gray); title("Mandrill Magnitude - formula");
subplot(2, 3, 5); imagesc(mandrill_phase); colormap(gray); title("Mandrill Phase - formula");

subplot(2, 3, 3); imagesc(log(1 + clown_magnitude)); colormap(gray); title("Clown Magnitude");
subplot(2, 3, 6); imagesc(clown_phase); colormap(gray); title("Clown Phase");


figure("name", "Swapped Mandrill & Clown frequency domain");
subplot(2, 2, 1); imagesc(log(1 + replaced_phase_fft_magnitude)); colormap(gray); title("Mandrill magnitude & Clown phase : Magnitude");
subplot(2, 2, 3); imagesc(replaced_phase_fft_phase); colormap(gray); title("Mandrill magnitude & Clown phase : Phase");
subplot(2,2,2); imagesc(log(1 + replaced_magnitude_fft_magnitude)); colormap(gray); title("Clown magnitude & Mandrill phase : Magnitude");
subplot(2,2,4); imagesc(replaced_magnitude_fft_phase); colormap(gray); title("Clown magnitude & Mandrill phase : Phase");


%% 4 Filtering in the frequency domain 
 %%   4.1 “Notch” Filter 
clear all; close all; clc;

complex = @(magnitude, phase) magnitude .* (cos(phase) + i .*sin(phase)); %Using the same formula as previous task.

pattern = im2double(imread("pattern.tif"));

%calculate magnitude and phase
pattern_fft = fftshift(fft2(pattern));
pattern_magnitude = abs(pattern_fft);
pattern_phase = angle(pattern_fft);


subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");

%I inspected the magnitude plot and saw that the central horizantal line
%and vertical line looks like its has a repetative pattern. Next I zoomed
%in to look at the central horizontal line only and could see a repetative pattern.
%I used the "data tips" tool to get the index of those pixels
%and made the conclution that they were 13 pixels apart, (in a few cases 12
%pixels apart). I did the same procedure for the vertical line and observed
%the same pattern, with 13 pixels apart.

%When I zoomed out again I noticed that it looked like the pattern could be
%repeated in the entire magnitude plot, forming a grid of 13 pixels apart
%in both vertical and horizontal direction. (But with a much lower strength
%then on the center lines)

%The notch filter I will design will be to remove these patterns. As the
%lectures demonstrated, an ideal notch filter in the frequency domain leads
%to ringning, instead a gaussian notch filter is used. The lectures also
%mentioned that the notch filter must be applied symetrical around the
%center otherwise unexpected behaviour can occur.

%First I will test to remove the pattern och the central horizontal and
%vertical line only and then I will test to remove the entrie grid pattern.


%1)Notch filter with gaussians placed every 13th pixel from the center on the
%central horizontal and vertical lines only. The notch filter will remove
%around every 13th pixel and keep the rest the same.

%Construct the gaussian to remove from the filter
kernel_size = 23;
kernel_std = kernel_size/5;

notch_reject = fspecial('gaussian', kernel_size, kernel_std);

%remapping the values to be stretched to the entire range [0, 1]
notch_reject = notch_reject - min(min(notch_reject));
notch_reject = notch_reject ./ max(max(notch_reject));

%Cunstruct the filter by passing everything through by adding just ones.
%Then remove the gaussian from this.
notch_central_lines = ones(size(pattern_magnitude));
center_row = round(size(pattern_magnitude, 1)/2) + 1;
center_column = round(size(pattern_magnitude, 2)/2) + 1;

%Remove on the central horizontal line (ignore the center)
for column_offset = 0:13:(center_column - 1)
        for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
            row_s = center_row + s;
            for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                if (column_offset ~= 0)
                    %symmetric around center 2 directions

                    column_t = center_column + column_offset + t; %Remove on the Right side of center
                    if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end

                    column_t = center_column - column_offset + t; %Remove on the Left side of center
                    if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end
                end
            end
        end
end

%Remove on the vertical horizontal line (ignore the center)
for row_offset = 0:13:(center_row - 1)
        for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
            for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                if(row_offset ~= 0)
                    column_t = center_column + t;
                    %symmetric around center 2 directions
                    
                    row_s = center_row + row_offset + s; %Remove Below center
                    if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end

                    row_s = center_row - row_offset + s; %Remove Above center
                     if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                     end
                end
            end
        end
end
notch_central_lines = max(0, notch_central_lines);%Make sure there are no negative values by setting them to 0.

subplot(2,2, 2); imagesc(notch_central_lines); colormap(gray); title("Notch filter center lines");
%The notch filter looks like what I wanted it too and I will test to apply it 
%to the magnitude of the pattern image.


pattern_magnitude_modified = notch_central_lines .* pattern_magnitude;
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified)); colormap(gray); title(["Pattern - Magnitude (Modified)", "Notch filter center lines"]);

%Finally the image is restored by combinding the modified magnitude with
%the phase and the performing inverse fourier transform.
pattern_fft_modified = complex(pattern_magnitude_modified, pattern_phase);
pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)", "Notch filter center lines"]);


%2) Now the same test with a grid filter (ignoring the center)
notch_grid = ones(size(pattern_magnitude));
for column_offset = 0:13:(center_column - 1)
    for row_offset = 0:13:(center_row - 1)
            for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                    
                    if (column_offset ~= 0 || row_offset ~= 0)
                        
                        %symmetric around center gives 4 directions.
                        
                        column_t = center_column + column_offset + t; %Remove Right of center
                        row_s = center_row + row_offset + s; %and Below center
                        if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                            if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                            end
                        end

                        if (row_offset ~= 0)%Only need to remove once on the center row
                            column_t = center_column + column_offset + t; %Remove Right of center
                            row_s = center_row - row_offset + s; % and Above center
                            if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                                if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                    notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                                end
                            end
                        end
                        
                        if (column_offset ~= 0)%Only need to remove once on the center column
                            column_t = center_column - column_offset + t;%Remove Left of center
                            row_s = center_row + row_offset + s; %and Below center
                            if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                                if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                    notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                                end
                            end

                            if (row_offset ~= 0) %Only need to remove once on the center row
                                column_t = center_column - column_offset + t; %Remove Left of center
                                row_s = center_row - row_offset + s; %and Above center
                                if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                                    if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                        notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                                    end
                                end
                            end
                        end
                        
                    end
                    
                end
            end
    end
end
notch_grid = max(0, notch_grid);%Make sure there are no negative values by setting them to 0.

figure; 
subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");
subplot(2,2,2); imagesc(notch_grid); colormap(gray); title("Notch filter grid");

%I will now test to apply the grid notch filter to the magnitude of the pattern image.
pattern_magnitude_modified_2 = notch_grid .* pattern_magnitude;
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified_2)); colormap(gray); title(["Pattern - Magnitude (Modified)","Notch filter grid"]);

%And see the result in the restored image.
pattern_fft_modified = complex(pattern_magnitude_modified_2, pattern_phase);
pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)","Notch filter grid"]);



%The grid notch filter removes the grid in the image better, but the
%central lines notch filter perserves the text better. I will experiment to
%see if a combination of them is better.

notch_3 = 2 .* notch_central_lines + 1 .* notch_grid;
pattern_magnitude_modified_3 = pattern_magnitude .* notch_3;

figure;
subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");
subplot(2,2,2); imagesc(notch_3); colormap(gray); title("Notch filter combined");
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified_3)); colormap(gray); title(["Pattern - Magnitude (Modified)", "Notch filter combined"]);

pattern_fft_modified = complex(pattern_magnitude_modified_3, pattern_phase);
pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)","Notch filter combined"]);


%The best case is the notch filter applied on the center lines only. 
%I guess that is to be expected concidering the examples shown on the
%lectures on fourier transform. An image with repeating lines in the
%horizontal axis gives a pattern on the center vertical axis in the
%magnitude plot in the frequency domain. And vice versa for a vertically 
%repeating pattern. The grid pattern in this image can be seen as vertically 
%repeating lines and horizontally repeating lines. The conclusion is then that 
%the pattern will most likely be visible on the horizontal and vertical center lines
%on the magnitude plot in the frequency domain, which seems to be the case.

%%    4.2 SÄPO task
clear all; close all; clc;

complex = @(magnitude, phase) magnitude .* (cos(phase) + i .*sin(phase));

car = im2double(imread("car_gray_corrupted.png"));

car_fft = fftshift(fft2(car));
car_magnitude = abs(car_fft);
car_phase = angle(car_fft);

figure;
subplot(2,2,1); imagesc(car); colormap(gray); title("Corupted image");
subplot(2,2,2); imagesc(log(1 + car_magnitude)); colormap(gray); title("Magnitude");


%By looking at the magnitude plot I can directly identify several bright
%spot, With the "data tips" tool I identify there position in the plot.
special_cases(1:2, 1) = [186; 216];
special_cases(1:2, end + 1) = [188; 218];

special_cases(1:2, end + 1) = [246; 156];
special_cases(1:2, end + 1) = [248; 158];

special_cases(1:2, end + 1) = [326; 296];
special_cases(1:2, end + 1) = [328; 298];

special_cases(1:2, end + 1) = [266; 356];
special_cases(1:2, end + 1) = [268; 358];
hold on; scatter(special_cases(2, :), special_cases(1, :), 'rO');

%First I will try to just remove these outliers
car_magnitude_modified = car_magnitude;
for i=1:length(special_cases(1, :))
    car_magnitude_modified(special_cases(1, i), special_cases(2,i)) = 0;
end
subplot(2,2,3); imagesc(log(1 + car_magnitude_modified)); colormap(gray); title("Magnitude - applied ideal notch");

%And display the restored image
car_restored_fft = complex(car_magnitude_modified, car_phase);
car_restored = real(ifft2(fftshift(car_restored_fft)));
subplot(2,2,4); imagesc(car_restored); colormap(gray); title("Restored image");


%Next I will try to remove with a gaussian around each outlier pixel.
%Removing with a gaussian in the same way as the notch filter task.
kernel_size = 17;
kernel_std = kernel_size/5;
notch_reject = fspecial('gaussian', kernel_size, kernel_std);
notch_reject = notch_reject - min(min(notch_reject));
notch_reject = notch_reject ./ max(max(notch_reject));

notch_filter = ones(size(car_magnitude));
for i = 1:length(special_cases(1,:))
    for s = -floor(kernel_size/2):floor(kernel_size/2)
        for t = -floor(kernel_size/2):floor(kernel_size/2)
            row_s = special_cases(1, i) + s;
            column_t = special_cases(2, i) + t;
            if(column_t >= 1 && column_t <= size(car_magnitude, 2))
                if(row_s >= 1 && row_s <= size(car_magnitude, 1))
                    notch_filter(row_s, column_t) = notch_filter(row_s, column_t) - notch_reject(s + round(kernel_size/2), t + round(kernel_size/2)); %Remove with the gaussian values
                end
            end
        end
    end
end
notch_filter = max(0, notch_filter);

%appliying the constucted filter with to gaussian placed at each outlier pixel.
car_magnitude_modified = car_magnitude .* notch_filter;


figure
subplot(2,2,1); imagesc(car); colormap(gray); title("Corrupted image");
subplot(2,2,2); imagesc(log(1 + car_magnitude)); colormap(gray); title("Magnitude");
hold on; scatter(special_cases(2, :), special_cases(1, :), 'rO');

subplot(2,2,3); imagesc(log(1 + car_magnitude_modified)); colormap(gray); title("Magnitude - applied gaussian notch");

%Demonstrating the result of the restored image
car_restored_fft = complex(car_magnitude_modified, car_phase);
car_restored = real(ifft2(fftshift(car_restored_fft)));
subplot(2,2,4); imagesc(car_restored); colormap(gray); title("Restored image");



%Both ideal notch filter and gaussian notch filter seem to give equall
%result in this case. It is also worth noting that the special cases found
%were symmetric around the center which could also be observed in the
%direction on the noise pattern in the corrupted image.







