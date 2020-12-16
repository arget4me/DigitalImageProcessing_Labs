%% Lab Assignements #3
%% 1 Recognition


%% 1.1 Feature detection

clear all; close all; clc;
cd 'images'
query_dir = dir('query/*.jpg');
database_dir = dir('database/*.jpg');
addpath('query');
addpath('database');
addpath('extra_test_images');
addpath('graffiti_images');
addpath('Oats');
cd ..

figure('name', 'NumOctaves = 3 vs NumOctaves = 1')
%NumOctaves' decide on the maximum sizes of the blobs

subplot(1, 2,1);
% read image file
I = imread('boat.tif');
% detect SURF features
pts = detectSURFFeatures(I);

% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;

subplot(1, 2, 2);
pts = detectSURFFeatures(I,'MetricThreshold',1000.0,'NumOctaves',1,'NumScaleLevels',4);
% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;

figure('name', 'NumScaleLevels = 3 vs NumScaleLevels = 6')
subplot(1, 2,1);
%'NumScaleLevels' decide on how the sizes of the blobs are distributed
%between smallest and largest size.
pts = detectSURFFeatures(I,'MetricThreshold',1000.0,'NumOctaves',3,'NumScaleLevels',3);
% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;

subplot(1, 2,2);
pts = detectSURFFeatures(I,'MetricThreshold',1000.0,'NumOctaves',3,'NumScaleLevels',6);
% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;


figure('name', 'MetricThreshold = 10 vs MetricThreshold = 20000')
subplot(1, 2,1);
%MetricThreshold decide on number of blobs found, lower means more blobs
%found
pts = detectSURFFeatures(I,'MetricThreshold',10.0,'NumOctaves',1,'NumScaleLevels',3);
% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;

subplot(1, 2,2);
pts = detectSURFFeatures(I,'MetricThreshold',20000.0,'NumOctaves',1,'NumScaleLevels',3);
% display image
imshow(I); hold on;
% display SURF points
plot(pts.selectStrongest(50)); hold off;


%%1.2 Feature extraction

% detect SURF features
pts = detectSURFFeatures(I);
% extract SURF descriptors
[feats, validPts] = extractFeatures(I,pts);
fprintf("number of validPts = %.1f\nnumber of pts = %.1f\n", length(validPts), length(pts))



%%1.3 Feature matching

% read image file
I1 = imread('graffiti1.png');
% convert to grayscale
Ig = rgb2gray(I1);
% detect SURF features
pts = detectSURFFeatures(Ig);
% extract SURF descriptors
[feats1, validPts1] = extractFeatures(Ig,pts);

% read image file
I2 = imread('graffiti3.png');
% convert to grayscale
Ig = rgb2gray(I2);
% detect SURF features
pts = detectSURFFeatures(Ig);
% extract SURF descriptors
[feats2, validPts2] = extractFeatures(Ig,pts);

% match feature sets from 2 images
indexPairs = matchFeatures(feats1, feats2);


% visualise matched features
matchedPoints1 = validPts1(indexPairs(:, 1));
matchedPoints2 = validPts2(indexPairs(:, 2));
figure('name', 'Graffiti - Match features'); showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage')


%%1.4 Image matching
database_images{length(database_dir)} = [];
database_features{length(database_dir), 2} = [];

query_images{length(query_dir)} = [];
query_features{length(query_dir), 2} = [];

%Load all database images and extract features.
for i = 1:length(database_dir)
    I = imread(database_dir(i).name);
    database_images{i} = I;
    I = rgb2gray(I);
    
    % detect SURF features
    pts = detectSURFFeatures(I);
    % extract SURF descriptors
    [feats, validPts] = extractFeatures(I, pts);
    
    database_features{i, 1} = feats;
    database_features{i, 2} = validPts;
end

%Load all query images and extract features
for i = 1:length(query_dir)
    I = imread(query_dir(i).name);
    query_images{i} = I;
    I = rgb2gray(I);
    
    % detect SURF features
    pts = detectSURFFeatures(I);
    % extract SURF descriptors
    [feats, validPts] = extractFeatures(I, pts);
    
    query_features{i, 1} = feats;
    query_features{i, 2} = validPts;
end

%Find the best match in the database of images for each query image
for i = 1:length(query_images)
    recognition(query_images{i}, query_features(i, :), database_images, database_features)
end
%% 2 Detection, segmentation and object statistics
%% 2.1 Scientific background and data
clear all; close all; clc;

%@QUESTION: Change structure inside matlab, or manually in folder??

cd images/Oats
    addpath('Gotala 50 corns')
    addpath('Lanna 50 corns')
    addpath('Multorp 50 corns')

    farms = {'Gotala', 'Lanna', 'Multorp'};
    all_images{1} = dir([farms{1} ' 50 corns/*.jpg']);
    all_images{2} = dir([farms{2} ' 50 corns/*.jpg']);
    all_images{3} = dir([farms{3} ' 50 corns/*.jpg']);
cd ../..

number_of_images = length(all_images{1}) + length(all_images{2}) +length(all_images{3});

%Collection row layout  {Image, Farm, Cultivation, ShootStage
cultivations = {'Belinda', 'Fatima', 'Symphony', 'Unkown'};
cultivations_search_words = [" bel ", " fat ", " symp "];
shoots = {'Huvudskott', 'Gronskott', 'Unkown'};
shoots_search_words = [" huv ", " grn "];

collection{number_of_images, 2} = [];
start_index_offset = 0;
for i = 1:length(all_images)
    for k = 1:length(all_images{i})
        current_index = start_index_offset + k;
        
        s = struct("file_name", '', "farm_index", 0, "cultivation_index", 0, "shoot_index", 0);
        s.file_name = all_images{i}(k).name;
        s.farm_index = i;
        s.cultivation_index = length(cultivations_search_words) + 1;
        s.shoot_index = length(shoots_search_words) + 1;
        
        for cultivation = 1:length(cultivations_search_words)
            if contains(lower(all_images{i}(k).name), cultivations_search_words(cultivation))
                s.cultivation_index = cultivation;
                break;
            end
        end
        
        for shoot = 1:length(shoots_search_words)
            if contains(lower(all_images{i}(k).name), shoots_search_words(shoot))
                s.shoot_index = shoot;
                break;
            end
        end
        
        collection{current_index, 1} = imrotate(imread(s.file_name), -90);
        collection{current_index, 2} = s;
        fprintf("'%s':\t%s\t%s\t%s\n", collection{current_index, 2}.file_name, farms{collection{current_index, 2}.farm_index}, cultivations{collection{current_index, 2}.cultivation_index}, shoots{collection{current_index, 2}.shoot_index});
        
    end
    
    start_index_offset = start_index_offset + length(all_images{i});
end


%% 2.2 Segmentation

index = randi(length(collection(:, 1)), 4);
%{
figure('name', 'Laplacian');
for i = 1:4
    img = imfilter(rgb2gray(collection{index(i), 1}), fspecial("laplacian"));%collection{index(i), 1}, fspecial("laplcian"))
    title_string = [farms{collection{index(i), 2}.farm_index},' ', cultivations{collection{index(i), 2}.cultivation_index},' ',shoots{collection{index(i), 2}.shoot_index}];
    subplot(2, 2, i); imshow(img); title(title_string);
end

figure('name', 'Sobel');
for i = 1:4
    img = imfilter(rgb2gray(collection{index(i), 1}), fspecial("sobel"));%collection{index(i), 1}, fspecial("laplcian"))
    title_string = [farms{collection{index(i), 2}.farm_index},' ', cultivations{collection{index(i), 2}.cultivation_index},' ',shoots{collection{index(i), 2}.shoot_index}];
    subplot(2, 2, i); imshow(img); title(title_string);
end
%}
%

for i = 1:1
     title_string = [farms{collection{index(i), 2}.farm_index},' ', cultivations{collection{index(i), 2}.cultivation_index},' ',shoots{collection{index(i), 2}.shoot_index}];
    img = im2double(rgb2gray(collection{index(i), 1}));
    figure('name', 'Normal');
    imshow(img); title(title_string);
    
    img = imfilter(img, fspecial('gaussian', 13, 7));
    img_bilateral = imbilatfilt(img, 0.1, 18, 'NeighborhoodSize', 9);
    
    figure('name', 'Bilateral filtering'); 
    imshow(img_bilateral); title(title_string);
    
    img_horizontal = abs(imfilter(img_bilateral, fspecial("sobel")));
    img_vertical = abs(imfilter(img_bilateral, fspecial("sobel")'));
    
    threshold = 0.2;
    
    threshold_index = find(img_vertical < threshold);
    img_vertical(threshold_index) = 0;
      threshold_index = find(img_vertical >= threshold);
    img_vertical(threshold_index) = 1;

     threshold_index = find(img_horizontal < threshold);
    img_horizontal(threshold_index) = 0;
     threshold_index = find(img_horizontal >= threshold);
    img_horizontal(threshold_index) = 1;
    
    
    
    
    img = img_vertical + img_horizontal;
    
   figure('name', 'Sobel edges'); 
   imshow(img); title(title_string);
end


%% 2.3 Object identification and statistisc collection

%% 2.4 From image coordinates to world size

%% 2.5 Answering the research questions