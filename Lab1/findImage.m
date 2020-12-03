function [outIndex, outImg] = findImage(img, collection)
%Find image in a collection of images by comparing histograms
%   [index, img] = findImage(img, collection)
%   img - Image to find in collection
%   collection - collection of images to search through
%   outIndex - index of best image match
%   outImg - image of best match

ref_hist = imGrey2histogram(img);

collection_hist = zeros(length(collection), length(ref_hist));
score = zeros(1, length(collection));

for i = 1:length(collection)
    collection_hist(i,:) = imGrey2histogram(collection{i});
    
    score(i) = sqrt((1/length(ref_hist)) * (collection_hist(i,:) - ref_hist)*(collection_hist(i,:) - ref_hist)');
end

[M,outIndex] = min(score);
outImg = collection{outIndex};
end

