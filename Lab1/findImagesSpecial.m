function [specialImgs, specialImgs_entropy] = findImagesSpecial(collection, normDist)
%Find special images in a collection of images by comparing image entropy
%   [specialImgs, specialImgs_entropy] = findImagesSpecial(collection, normDist)
%   collection - collection of images to search through
%   normDist - normal distribution of image entropy from training data 
%   specialImgs - index of special images in collection
%   specialImgs_entropy - entropy for the special images found

%calculate entropy for test set
entropy = zeros(1, length(collection));
for i = 1:length(collection)
    entropy(1, i) = imGreyEntropy(collection{i});
end

%mark an image as "special" if its image entropy is outside of 2 standard diviations
specialImgs = find(abs(entropy - normDist.mu) > 2 * normDist.sigma);
specialImgs_entropy = entropy(specialImgs);

end

