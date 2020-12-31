function [statistics] = image_statistics(img, display_image)
%IMAGE_STATISTICS Compute the statistics for the image
%   img - orignial rgb image.
%   display_image - logic variable, if true then display image mask.
%	statistics - vector with image statistics in the following order: 
%   statistics = [number of kernels; 
%      average area; median area;
%      average minAxisLength; median minAxisLength;
%      average maxAxisLength; median maxAxisLength;
%      average r; median r;
%      average g; median g;
%      average b; median b;
%      average h; median h;
%      average s; median s;
%      average v; median v;
%   ]

    PIXELS_PER_UNIT = 52;
    SI_UNIT_RATIO = 1/200; %[M] 1:200
    MM_PER_M = 1000;
    SIZE_PER_PIXEL =  SI_UNIT_RATIO * MM_PER_M / PIXELS_PER_UNIT; %[mm]
    PIXEL_AREA_PER_MM2 = SIZE_PER_PIXEL*SIZE_PER_PIXEL; %conversion ratio from pixel area to milimeter area    

    filter_mask = get_filter_mask(rgb2gray(img));
    num_kernels = max(bwlabel(filter_mask), [], 'all');
    
    %Compute regions in the binary mask and present statistics for Area, MinorAxisLength and MajorAxisLength
    cc = bwconncomp(filter_mask);
    
    areas = regionprops(cc, 'Area');
    areas = PIXEL_AREA_PER_MM2 .* reshape(struct2array(areas), [], numel(fieldnames(areas)));%Convert from struct to vector

    min_axis = regionprops(cc, 'MinorAxisLength');
    min_axis = SIZE_PER_PIXEL .* reshape(struct2array(min_axis), [], numel(fieldnames(min_axis)));%Convert from struct to vector

    max_axis = regionprops(cc, 'MajorAxisLength');
    max_axis = SIZE_PER_PIXEL .* reshape(struct2array(max_axis), [], numel(fieldnames(max_axis)));%Convert from struct to vector

    
    %Multiply the current image with the binary mask, then convert the masked RGB image to HSV.
    filter_mask_rgb = filter_mask .* img;
    filter_mask_hsv = rgb2hsv(filter_mask_rgb);
    
    %Separate rgb and hsv values into separtate row vector for simpler use of the functions: mean(), median() and std()
    r = filter_mask_rgb(:, :, 1); r = r(:);
    g = filter_mask_rgb(:, :, 2); g = g(:);
    b = filter_mask_rgb(:, :, 3); b = b(:);

    h = filter_mask_hsv(:, :, 1); h = h(:);
    s = filter_mask_hsv(:, :, 2); s = s(:);
    v = filter_mask_hsv(:, :, 3); v = v(:);
    
    %Don't calculate statistics for parts outside the binary mask. Outside meaning where the binary mask is equal 0.
    remove_zeros = find(filter_mask == 0);
    r(remove_zeros) = [];
    g(remove_zeros) = [];
    b(remove_zeros) = [];
    h(remove_zeros) = [];
    s(remove_zeros) = [];
    v(remove_zeros) = [];
    
    if (display_image)
        figure
        imshow(0.3 .* img + 0.7 .* (filter_mask .* img));
    end
    
	statistics = [num_kernels; 
        mean(areas); median(areas);
        mean(min_axis); median(min_axis);
        mean(max_axis); median(max_axis);
        mean(r); median(r);
        mean(g); median(g);
        mean(b); median(b);
        mean(h); median(h);
        mean(s); median(s);
        mean(v); median(v);
	];
end

