function [img_out] = get_filter_mask(img)
%GET_FILTER_MASK strech values to cover full range of 0 to 1, then median filter, then bilateral filter, then thresholding, then remove small
%and large objects. Returns a binary filter mask image.
%   img - input image
%   img_out - output image, binary filter mask.


img = (img - min(img, [], 'all')) ./ ( max(img, [], 'all') - min(img, [], 'all') );%Streatch values
img = medfilt2(img, [13 13]);%Median filter
img = imbilatfilt(img, 0.3, 8, 'NeighborhoodSize', 9);% Bilateral filter

%thresholding
img = img > 0.62;


cc = bwconncomp(img);
areas = regionprops(cc, 'Area');
x = reshape(struct2array(areas), [], numel(fieldnames(areas)));
img_remove = zeros(size(img));

%Remove small areas
remove_areas_small = find(x < 600);
for i = 1:length(remove_areas_small)
    current_region = cc.PixelIdxList{remove_areas_small(i)}; 

    for k = 1:length(current_region)
        img_remove(current_region(k)) = 1;
    end
end

%Remove large objects, Assume that they are rectangulare, remove entire
%area inside the min and max x,y positions.
 remove_areas_big = find(x > 8000);
for i = 1:length(remove_areas_big)
    current_region = cc.PixelIdxList{remove_areas_big(i)}; 

    y_min = size(img, 1);
    y_max = 0;
    x_min = size(img, 2);
    x_max = 0;
    for k = 1:length(current_region)
        [row, col] = ind2sub(size(img), current_region(k));

        if row < y_min
            y_min = row;
        end

        if row > y_max
            y_max = row;
        end

        if col < x_min
            x_min = col;
        end

        if col > x_max
            x_max = col;
        end
    end


    %Rectangular removing
    for col = x_min:x_max
        for row = y_min:y_max
            img_remove(row, col) = 1;
        end
    end
end


img = img - img_remove;
img_out = img > 0.0; %to remove negative parts

end

