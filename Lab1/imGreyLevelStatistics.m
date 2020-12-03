function [imgOut] = imGreyLevelStatistics(img)
% Computes the statistics about the difference in grey levels between adjacent pixels
%   [imgOut] = imGreyLevelStatistics(img, grey_levels)
%	img - image to compute grey level statistics on
%   imgOut - output image with each pixel containing statistics value

imgOut = img;
[num_rows, num_columns] = size(img);
for column = 1:num_columns
    for row = 1:num_rows
        differense = 0;
        weight = 1 / 8; % Not 1/9 since the center is unwanted. (will contribute 0 to the sum)
        
        %sum up average neighbor differense
        for s = -1:1
           for t = -1:1
                a = column + s; b = row + t;
                if((a <= 0) || (a > num_columns) )
                    a = column - s; %mirror at edge
                end
                if((b <= 0) || (b > num_rows))
                    b = row - t; %mirror at edge
                end
                
                differense = differense + weight * abs(double(img(row, column)) - double(img(b,a)));
           end
        end
        
        imgOut(row, column) = uint8(differense);
    end
end

%Uncomment to see difference better
img_max = double(max(imgOut));
imgOut = uint8(double(imgOut) ./ img_max .* 255);

end
