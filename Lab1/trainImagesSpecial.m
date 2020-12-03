function [normDist] = trainImagesSpecial(collection)
%Calculate normal distribution of image entropy from traning data
%   [normDist] = trainImagesSpecial(collection)
%   collection - collection of images to train model with
%   normDist - normal distribution of image entropy from training data

%calculate entropy for every image to train model
entropy = zeros(1, length(collection));
for i = 1:length(collection)
    entropy(1, i) = imGreyEntropy(collection{i});
end
normDist = fitdist(entropy', 'Normal');

end
