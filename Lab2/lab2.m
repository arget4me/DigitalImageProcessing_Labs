%% Lab Assignements #2
%% 1 Setup
clear all; close all; clc;

addpath('extra_test_images')
tif_files = dir("extra_test_images/*.tif");
png_files = dir("extra_test_images/*.png");
image_files = [tif_files;png_files]

num_rows = floor(sqrt(length(image_files)));
num_columns = ceil(sqrt(length(image_files)));
figure
for i = 1:length(image_files)
    subplot(num_rows, num_columns, i);
    I = imread(image_files(i).name);
    imshow(I); title(image_files(i).name, 'Interpreter', 'none');
end

%% 1  Linear Spatial Filtering
%%    1.1 Convolution Operator
clear all; close all; clc;
gauss_mask = fspecial('gaussian', 3, 0.849);

tif_files = dir("extra_test_images/*.tif");
png_files = dir("extra_test_images/*.png");
image_files = [tif_files;png_files];

kernel = gauss_mask

num_rows = length(image_files);
num_columns = 3;
figure
for i = 1:length(image_files)
    I = im2double(imread(image_files(i).name));
    
    subplot(num_rows, num_columns, num_columns * i - 2);
    img_out = convolution_operator(I, kernel); %mirror at border
    imshow(img_out, []); title('Gaussian', 'Interpreter', 'none');
    
    subplot(num_rows, num_columns, num_columns * i - 1);
    img_filter = imfilter(I, kernel, 'conv', 'symmetric'); %mirror at border
    imshow(img_filter, []); title('Imfilter', 'Interpreter', 'none');
    
    subplot(num_rows, num_columns, num_columns * i - 0);
    imshow(img_filter - img_out, []); title('Subtract: Gaussian & Imfilter', 'Interpreter', 'none');
end

 %%   1.2 Smoothing Filters
clear all; close all; clc;

I = im2double(imread("clown.tif"));


figure
subplot(2, 4, 1);
imshow(I);title('Original', 'Interpreter', 'none');

subplot(2, 4, 2);
img_mean_3 = convolution_operator(I, fspecial('average', 3));
imshow(img_mean_3, []); title('Mean 3x3', 'Interpreter', 'none');

subplot(2, 4, 3);
img_mean_5 = convolution_operator(I, fspecial('average', 5));
imshow(img_mean_5, []); title('Mean 5x5', 'Interpreter', 'none');

subplot(2, 4, 4);
img_mean_9 = convolution_operator(I, fspecial('average', 9));
imshow(img_mean_9, []); title('Mean 9x9', 'Interpreter', 'none');

subplot(2, 4, 5);
imshow(I);title('Original', 'Interpreter', 'none');

sigma = 0.849;
subplot(2, 4, 6);
img_gauss_3 = convolution_operator(I, fspecial('gaussian', 3, sigma));
imshow(img_gauss_3, []); title('Gaussian 3x3', 'Interpreter', 'none');

subplot(2, 4, 7);
img_gauss_5 = convolution_operator(I, fspecial('gaussian', 5, sigma));
imshow(img_gauss_5, []); title('Gaussian 5x5', 'Interpreter', 'none');

subplot(2, 4, 8);
imshow(img_gauss_3 - img_gauss_5, []); title('Subtract: 3x3 & 5x5', 'Interpreter', 'none');



%%    1.3 Edge Filters
clear all; close all; clc;

I = im2double(imread("clown.tif"));

%%%%%%%% @NOTE change to not use fspecial, insted use from the lectures and
%%%%%%%% describe the relation between the gradient and laplacian.. (first vs 2nd derivative)

figure
subplot(1, 2, 1);
kernel = fspecial('sobel')
img_out = convolution_operator(I, kernel); %mirror at border
imshow(img_out, []); title('Sobel', 'Interpreter', 'none');

subplot(1, 2, 2);
kernel = fspecial('laplacian')
img_out = convolution_operator(I, kernel); %mirror at border
imshow(img_out, []); title('Laplacian', 'Interpreter', 'none');


%% 2 Bilateral Filtering
clear all; close all; clc;

sigma_range = 6; %[2,  6, 18]; 
%@Note: if range_sigma is larger than the kernel size it will have little
%affect due to the normalization factor, Wp, rescaling the values used. In
%other words the weights in the kernel is rescaled to sum up to 1. If
%range_sigma is larger then the kernel size then the assumption is that in 
%just one standardeviation the kernel_size is exceeded and can't affect the result. 
%What will tend to happen with a much larger range_sigma then the kernel_size is 
%that only a small part around the mean is used and results in a filter similar to the 
%mean filter instead of a gaussian (For range only. the intensity is a gaussian 
%with sigma_domain as standard deviation)

sigma_domain = 0.1; %[0.1, 0.25, 10];
kernel_size = 9;

subplot(1, 3, 1); boat = im2double(imread("boat.tif")); imshow(boat); title("Boat Original");
subplot(1, 3, 2); clown = im2double(imread("clown.tif")); imshow(clown); title("Clown Original");
subplot(1, 3, 3); mandrill = im2double(imread("mandrill.tif")); imshow(mandrill); title("Mandrill Original");

boats = figure; clowns = figure; mandrills = figure;
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
    the in the same section and present the result from the at the same
    time.
%}


%  3.1 Phase and Magnitude -----------------------------------------------------------------------------------------------------------

clear all; close all; clc;

mandrill = im2double(imread("mandrill.tif")); mandrill = rgb2gray(mandrill);

mandrill_fft = fftshift(fft2(mandrill));
mandrill_magnitude_matlab = abs(mandrill_fft);
mandrill_phase_matlab = angle(mandrill_fft);


%  3.2 Phase vs Magnitude -------------------------------------------------------------------------------------------------------------
%generate a random complex number
z1 = (2 * rand() - 1) + (2 * rand() * i - i);

%magnitude: Matlab function abs(complex_number)
%abs(complex_number) = sqrt(Re(complex_number)^2 + Im(complex_number)^2)
magnitude = @(complex_number) sqrt(real(complex_number).^2 + imag(complex_number).^2)
r = magnitude(z1);
diff_abs = r - abs(z1);

%phase angle: Matlab function angle(complex_number)
%angle(complex_number) = atan(Im(complex_number) / Re(complex_number))
phase = @(complex_number) atan2(imag(complex_number), real(complex_number))
theta = phase(z1);
diff_angle = theta - angle(z1);


%eulers formula exp(-j * phi) = cos(phi) + i*sin(phi)
%using magnitude and phase
%complex_number = magnitude * (cos(phase) + i * sin(phase))
complex = @(magnitude, phase) magnitude .* (cos(phase) + i .*sin(phase))
z2 = complex(r, theta);
diff_complex = magnitude(z2 - z1);

fprintf("Complex number z = %f + %fi\n", real(z1), imag(z1));
fprintf("r = magnitude(z1) = %f. Difference between r and MATLAB function abs(z1) = %f\n", r, diff_abs);
fprintf("theta = phase(z1) = %f. Difference between theta and MATLAB function angle(z1) = %f\n", theta, diff_angle);
fprintf("z2 = complex(r, theta) = %f + %fi. Difference between z2 and original z1 = %f\n", real(z2), imag(z2), diff_complex);

% calculate magnitude and phase for "mandrill.tif" with formulas
mandrill_magnitude = magnitude(mandrill_fft);
mandrill_phase = phase(mandrill_fft);

%load clown image and calculate phase. 
clown = im2double(imread("clown.tif"));
clown_fft = fftshift(fft2(clown));
clown_magnitude = magnitude(clown_fft);
clown_phase = phase(clown_fft);

%Switch phase and magnitude between "clown.tif" and "mandril.tif" and
%perform inverse fourier transform
replaced_phase = complex(mandrill_magnitude, clown_phase);
replaced_phase_result = real(ifft2(fftshift(replaced_phase)));

replaced_magnitude = complex(clown_magnitude, mandrill_phase);
replaced_magnitude_result = real(ifft2(fftshift(replaced_magnitude)));

%Calculate magnitude and phase for the new images
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

complex = @(magnitude, phase) magnitude .* (cos(phase) + i .*sin(phase));

pattern = im2double(imread("pattern.tif"));

pattern_fft = fftshift(fft2(pattern));
pattern_magnitude = abs(pattern_fft);
pattern_phase = angle(pattern_fft);

figure; 
subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");

%I inspected the magnitude plot and saw that the central horizantal line
%and vertical line looks like its has a repetative pattern. Next I zoomed
%in to look at the central horizontal line only and could see a repetative pattern.
%I used the "data tips" tool to get the index of the pixels those pixels
%and made the conclution that they were 13 pixels apart, (in a few cases 12
%pixels apart). I did the same procedure for the vertical line and observed
%the same pattern, with 13 pixels apart.

%When I zoomed out again I noticed that it looked like the pattern was
%repeated in the entire magnitude plot, forming a grid of 13 pixels apart
%in both vertical and horizontal direction.

%The notch filter I will design will be to remove these patterns. As the
%lectures demonstrated, an ideal notch filter in the frequency domain leads
%to ringning, instead a gaussian notch filter is used. The lectures also
%mentioned that the notch filter must be applied symetrical around the
%center otherwise unexpected behaviour can occur.

%First I will test to remove the pattern och the central horizontal and
%vertical line only and then I will test to remove the entrie grid pattern.

%Notch filter with gaussians placed every 13th pixel from the center on the
%central horizontal and vertical lines only. The notch filter will remove
%around every 13th pixel and keep the rest the same.

kernel_size = 23;
kernel_std = kernel_size/5;
notch_reject = fspecial('gaussian', kernel_size, kernel_std);
notch_reject = notch_reject - min(min(notch_reject));
notch_reject = notch_reject ./ max(max(notch_reject));

notch_central_lines = ones(size(pattern_magnitude)); %Keep other parts the same
center_row = round(size(pattern_magnitude, 1)/2) + 1;
center_column = round(size(pattern_magnitude, 2)/2) + 1;

%central horizontal line
for column_offset = 0:13:(center_column - 1)
        for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
            row_s = center_row + s;
            for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                if (column_offset ~= 0)
                    %symmetric around center

                    column_t = center_column + column_offset + t; %Right side of center
                    if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end

                    column_t = center_column - column_offset + t; %Left side of center
                    if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end
                end
            end
        end
end

%central vertical line
for row_offset = 0:13:(center_row - 1)
        for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
            for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                if(row_offset ~= 0)
                    column_t = center_column + t;
                    %symmetric around center
                    row_s = center_row + row_offset + s; %Below center
                    if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                    end

                    row_s = center_row - row_offset + s; %Above center
                     if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                        notch_central_lines(row_s, column_t) = notch_central_lines(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                     end
                end
            end
        end
end
notch_central_lines = max(0, notch_central_lines);
subplot(2,2, 2); imagesc(notch_central_lines); colormap(gray); title("Notch filter center lines");


%The notch filter looks like what I wanted it too and I will test to apply it 
%to the magnitude of the pattern image.

pattern_magnitude_modified = notch_central_lines .* pattern_magnitude;
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified)); colormap(gray); title(["Pattern - Magnitude (Modified)", "Notch filter center lines"]);

pattern_fft_modified = complex(pattern_magnitude_modified, pattern_phase);

pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)", "Notch filter center lines"]);

%Now to grid filter
notch_grid = ones(size(pattern_magnitude));
for column_offset = 0:13:(center_column - 1)
    for row_offset = 0:13:(center_row - 1)
            for s = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                for t = -floor(size(notch_reject,1)/2):floor(size(notch_reject,1)/2)
                    if (column_offset ~= 0 || row_offset ~= 0)
                        
                        %symmetric around center
                        
                        column_t = center_column + column_offset + t; %Right of center
                        row_s = center_row + row_offset + s; %Below center
                        if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                            if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                            end
                        end

                        if (row_offset ~= 0)
                            column_t = center_column + column_offset + t; %Right of center
                            row_s = center_row - row_offset + s; %Above center
                            if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                                if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                    notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                                end
                            end
                        end
                        
                        if (column_offset ~= 0)
                            column_t = center_column - column_offset + t;%Left of center
                            row_s = center_row + row_offset + s; %Below center
                            if (row_s >= 1 && row_s <= size(pattern_magnitude, 1))
                                if (column_t >= 1 && column_t <= size(pattern_magnitude, 2))
                                    notch_grid(row_s, column_t) = notch_grid(row_s, column_t) - notch_reject(s + round(size(notch_reject,1)/2), t + round(size(notch_reject,1)/2)); %Remove with the gaussian values
                                end
                            end

                            if (row_offset ~= 0)
                                column_t = center_column - column_offset + t; %Left of center
                                row_s = center_row - row_offset + s; %Above center
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
notch_grid = max(0, notch_grid);
figure; 
subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");
subplot(2,2,2); imagesc(notch_grid); colormap(gray); title("Notch filter grid");

%I will now test to apply the grid notch filter to the magnitude of the pattern image.

pattern_magnitude_modified_2 = notch_grid .* pattern_magnitude;
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified_2)); colormap(gray); title(["Pattern - Magnitude (Modified)","Notch filter grid"]);

pattern_fft_modified = complex(pattern_magnitude_modified_2, pattern_phase);

pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)","Notch filter grid"]);



%The grid notch filter removes the grid in the image better, but the
%central lines notch filter perserves the text better. I will experiment to
%see if a combination of them is better.

notch_3 = ones(size(pattern_magnitude)) + 4 .* notch_central_lines + 1 .* notch_grid;
notch_3 = notch_3 - min(min(notch_3)); notch_3 = notch_3 ./ max(max(notch_3));

pattern_magnitude_modified_3 = pattern_magnitude .* notch_3;

figure;
subplot(2,2,1); imagesc( log(1 + pattern_magnitude)); colormap(gray); title("Pattern - Magnitude");
subplot(2,2,2); imagesc(notch_3); colormap(gray); title("Notch filter combined");
subplot(2,2,3); imagesc(log(1 + pattern_magnitude_modified_3)); colormap(gray); title(["Pattern - Magnitude (Modified)", "Notch filter combined"]);

pattern_fft_modified = complex(pattern_magnitude_modified_3, pattern_phase);

pattern_modified = real(ifft2(fftshift(pattern_fft_modified)));
subplot(2,2,4); imagesc(pattern_modified); colormap(gray); title(["Pattern (Modified)","Notch filter combined"]);

%The best case is the natch filter applied on the center lines only. 
%I guess that is to be expected concidering the examples shown in the
%lectures on fourier transform. An image with repeating lines in the
%horizontal axis gives a pattern on the center vertical axis in the
%frequency domain. And vice versa for a vertically repeating pattern.
%The grid pattern in this image can be seen as vertically repeating lines
%and horizontally repeating lines. The conclusion is then that the pattern
%will most likely be visiable on the horizontal and vertical center lines
%in the frequency domain, which seems to be the case.

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
car_magnitude_modified = car_magnitude;
%First I will try to just remove these outliers
for i=1:length(special_cases(1, :))
    car_magnitude_modified(special_cases(1, i), special_cases(2,i)) = 0;
end

%magnitude_edge = imfilter(log(1 + car_magnitude), fspecial("laplacian"))

subplot(2,2,3); imagesc(log(1 + car_magnitude_modified)); colormap(gray); title("Magnitude - applied ideal notch");

car_restored_fft = complex(car_magnitude_modified, car_phase);
car_restored = real(ifft2(fftshift(car_restored_fft)));

subplot(2,2,4); imagesc(car_restored); colormap(gray); title("Restored image");


%Next I will try to remove the with a gaussian around each outlier pixel

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


car_magnitude_modified = car_magnitude .* notch_filter;

figure
subplot(2,2,1); imagesc(car); colormap(gray); title("Corrupted image");
subplot(2,2,2); imagesc(log(1 + car_magnitude)); colormap(gray); title("Magnitude");

subplot(2,2,3); imagesc(log(1 + car_magnitude_modified)); colormap(gray); title("Magnitude - applied gaussian notch");

car_restored_fft = complex(car_magnitude_modified, car_phase);
car_restored = real(ifft2(fftshift(car_restored_fft)));

subplot(2,2,4); imagesc(car_restored); colormap(gray); title("Restored image");




%Both ideal notch filter and gaussian notch filter seem to give equall
%result in this case. It is also worth noting that the special cases found
%were symmetric around the center which could also be observed in the
%direction on the noise patter in the corrupted image.







