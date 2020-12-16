function recognition(query_image, query_features, database_images, database_features)
%RECOGNITION Find the image in tha database of images with the most number of matched features.
%   recognition(query_image, query_features, database_images, database_features)
%   query_image - The query image to find the best match for. Can be rbg or gray-scale.
%   query_features - a [1 x 2] collection. Cell 1 holds features found using
%   extractFeatures on the query image. Cell 2 holds SURFPoints struct for
%   the feature.
%   database_images - a [n x 1] collection of database images, each cell holds an image, rgb or gray-scale.
%   database_features - a [n x 2] collection of features and SURFPoints for the database images,
%       each cell in column 1 holds the features found using extractFeatures on the image
%       in the same position in the database_images collection.
%       each cell in column 2 holds the SURFPoints struct for the features found using extractFeatures on the image
%       in the same position in the database_images collection.

best_match_count = 0;
best_match_index = 0;
best_match_pairs{2} = [];

for i = 1:length(database_images)

    indexPairs = matchFeatures(query_features{1}, database_features{i, 1});
    if length(indexPairs(:, 1)) > best_match_count
        %new best match
        best_match_count = length(indexPairs(:, 1));
        best_match_index = i;
        best_match_pairs{2} = [];
        best_match_pairs{1} = query_features{2}(indexPairs(:, 1));
        best_match_pairs{2} = database_features{i, 2}(indexPairs(:, 2));
    end
end

if best_match_index > 0
    figure('name', 'Recognition'); showMatchedFeatures(query_image, database_images{best_match_index}, best_match_pairs{:, 1}, best_match_pairs{:, 2}, 'montage')
end

