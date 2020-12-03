function [imgOut] = imReduceGreyLevel(img, grey_levels)
% Reduce the grey levels of a grey-scale image.
%   [imgOut] = imReduceGreyLevel(img, grey_levels)
%	img - image to change grey leves
%   grey_levels - number of wanted grey_levels in range [2-256]
%   imgOut - output image with wanted grey levels
if(grey_levels > 256)
    grey_levels = 256;
else
    if(grey_levels < 2)
        grey_levels = 2;
    end
end

%The first grey level start at 0, which leaves (grey_levels - 1)values for the remaining levels.
levels_color_step = idivide(255, uint8(grey_levels - 1), 'floor'); 
levels_boundary = idivide(255, uint8(grey_levels), 'floor');
imgOut = idivide(img, levels_boundary, 'floor') * levels_color_step;

end