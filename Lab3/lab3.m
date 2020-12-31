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
        
        s = struct("file_name", all_images{i}(k).name, "farm_index", i, "cultivation_index", length(cultivations), "shoot_index", length(shoots), "pair_tag", 'no-tag');
        
        %Find cultivation type (overwrites default type if a new type is found in the name)
        for cultivation = 1:length(cultivations_search_words)
            if contains(lower(s.file_name), cultivations_search_words(cultivation))
                s.cultivation_index = cultivation;
                break;
            end
        end
        
        %Find shoot type (overwrites default type if a new type is found in the name)
        for shoot = 1:length(shoots_search_words)
            if contains(lower(s.file_name), shoots_search_words(shoot))
                s.shoot_index = shoot;
                break;
            end
        end
        
        %Save tag name. A substing is a tag i it starts with a number and doesn't contain '.JPG' (overwrites default type if a new type is found in the name)
        parts = split(upper(s.file_name));
        for p = 1:length(parts)
            str = parts{p};
            if(~contains(str, '.JPG'))
                if(ismember(str(1), '0123456789'))
                    s.pair_tag = str;
                    break;
                end
            end
        end
        
        %If there is not a cultivation set and the farm is Multorp and there exists a
        %pair_tag and if the pair tag only contains numbers, then the
        %cultivation can be extracted.
        %{
            Samples 50-59 are Belinda
            Samples 63-72 are Fatima
            Samples 74-83 are Symphony
        %index layout --> cultivations = {'Belinda', 'Fatima', 'Symphony', 'Unkown'};
        %}
        if(s.cultivation_index == length(cultivations) && s.farm_index == length(farms) && ~strcmp(s.pair_tag, 'no-tag'))
            if (all(ismember(s.pair_tag, '0123456789')))
                num = str2num(s.pair_tag);
                
                if (num >= 50 && num <= 59)
                    s.cultivation_index = 1;
                elseif (num >= 63 && num <= 72)
                    s.cultivation_index = 2;
                elseif (num >= 74 && num <= 83)
                    s.cultivation_index = 3;
                end
                
            end
        end
        
        collection{current_index, 1} = imrotate(imread(s.file_name), -90);%Rotate all images 90 degrees clockwise
        collection{current_index, 2} = s; %Store the tags for the current image.
        fprintf("'%s': \t %s \t %s \t %s \t %s \n", upper(collection{current_index, 2}.file_name), upper(farms{collection{current_index, 2}.farm_index}), upper(cultivations{collection{current_index, 2}.cultivation_index}), upper(shoots{collection{current_index, 2}.shoot_index}), upper(collection{current_index, 2}.pair_tag));
        
    end
    
    start_index_offset = start_index_offset + length(all_images{i});
end


%% 2.2 Segmentation
close all; clc;
%issue_img = [15, 55, 57, 47, 40, 33, 59]
fprintf("\nCurrent index:");
for index = 1:length(collection(:, 1))
    
    if mod(index - 1, 10) == 0
        fprintf("\n");
    end
    fprintf("%d\t", index);
    title_string = [num2str(index), ': ', farms{collection{index, 2}.farm_index},' ', cultivations{collection{index, 2}.cultivation_index},' ',shoots{collection{index, 2}.shoot_index}, ' ', collection{current_index, 2}.pair_tag];

    
    figure('name', title_string);
    subplot(1, 2, 1); imshow(collection{index, 1})

    img = im2double(rgb2gray(collection{index, 1}));
    filter_mask = get_filter_mask(img);
    
    subplot(1, 2, 2); imshow(filter_mask)
end

%% 2.3 Object identification and statistisc collection
close all; clc;

random_index = randi(number_of_images);
random_index_pair = 0;

%find the respective pair for the current random image: (Resulting in a pair of Huvudskott and Gronskott)
for i = 1:number_of_images
    check_farm_index = (collection{i, 2}.farm_index == collection{random_index, 2}.farm_index);
    check_cultivation_index = (collection{i, 2}.cultivation_index == collection{random_index, 2}.cultivation_index);
    check_shoot_index = (collection{i, 2}.shoot_index ~= collection{random_index, 2}.shoot_index);
    check_pair_tag = strcmp(collection{i, 2}.pair_tag, collection{random_index, 2}.pair_tag);
    
    if(check_farm_index && check_cultivation_index && check_shoot_index && check_pair_tag)
        random_index_pair = i;
        break;
    end
end

%Present statistics for the pair of Huvudskott and Gronskott
for index = [random_index, random_index_pair]
    %Print index and tags for current image
    title_string = [num2str(index), ': ', farms{collection{index, 2}.farm_index},' ', cultivations{collection{index, 2}.cultivation_index},' ',shoots{collection{index, 2}.shoot_index}, ' ', collection{current_index, 2}.pair_tag];
    fprintf("%s\n", title_string);
    
    %Compute the binary mask for the image.
    img = im2double(collection{index, 1});
    filter_mask = get_filter_mask(rgb2gray(img));
    
    %Compute regions in the binary mask and present statistics for Area, MinorAxisLength and MajorAxisLength
    cc = bwconncomp(filter_mask);
    
    areas = regionprops(cc, 'Area');
    areas = reshape(struct2array(areas), [], numel(fieldnames(areas)));%Convert from struct to vector
    fprintf("Area: average = %.3fpx^2, median = %.3fpx^2, std = %.3fpx^2\n", mean(areas), median(areas), std(areas))

    min_axis = regionprops(cc, 'MinorAxisLength');
    min_axis = reshape(struct2array(min_axis), [], numel(fieldnames(min_axis)));%Convert from struct to vector
    fprintf("MinorAxisLength: average = %.3fpx, median = %.3fpx, std = %.3fpx\n", mean(min_axis), median(min_axis), std(min_axis))

    max_axis = regionprops(cc, 'MajorAxisLength');
    max_axis = reshape(struct2array(max_axis), [], numel(fieldnames(max_axis)));%Convert from struct to vector
    fprintf("MajorAxisLength: average = %.3fpx, median = %.3fpx, std = %.3fpx\n\n", mean(max_axis), median(max_axis), std(max_axis))

    
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

    %Present statistics for RGB and HSV
    fprintf("r: average = %.3f, median = %.3f, std = %.3f\n", mean(r), median(r), std(r))
    fprintf("g: average = %.3f, median = %.3f, std = %.3f\n", mean(g), median(g), std(g))
    fprintf("b: average = %.3f, median = %.3f, std = %.3f\n\n", mean(b), median(b), std(b))
    fprintf("h: average = %.3f, median = %.3f, std = %.3f\n", mean(h), median(h), std(h))
    fprintf("s: average = %.3f, median = %.3f, std = %.3f\n", mean(s), median(s), std(s))
    fprintf("v: average = %.3f, median = %.3f, std = %.3f\n\n\n", mean(v), median(v), std(v))
    
    
    %Display current image and the image multiplied by the binary mask.
    figure('name', title_string);
    subplot(1, 2, 1); imshow(img);
    subplot(1, 2, 2); imshow(filter_mask_rgb)
end



%% 2.4 From image coordinates to world size

close all; clc;

random_index = randi(number_of_images);
random_index_pair = 0;

%find the respective pair for the current random image: (Resulting in a pair of Huvudskott and Gronskott)
for i = 1:number_of_images
    check_farm_index = (collection{i, 2}.farm_index == collection{random_index, 2}.farm_index);
    check_cultivation_index = (collection{i, 2}.cultivation_index == collection{random_index, 2}.cultivation_index);
    check_shoot_index = (collection{i, 2}.shoot_index ~= collection{random_index, 2}.shoot_index);
    check_pair_tag = strcmp(collection{i, 2}.pair_tag, collection{random_index, 2}.pair_tag);
    
    if(check_farm_index && check_cultivation_index && check_shoot_index && check_pair_tag)
        random_index_pair = i;
        break;
    end
end

%Present statistics for the pair of Huvudskott and Gronskott
for index = [random_index, random_index_pair]
    %Print index and tags for current image
    title_string = [num2str(index), ': ', farms{collection{index, 2}.farm_index},' ', cultivations{collection{index, 2}.cultivation_index},' ',shoots{collection{index, 2}.shoot_index}, ' ', collection{current_index, 2}.pair_tag];
    fprintf("%s\n", title_string);
    
    %Compute the binary mask for the image.
    img = im2double(collection{index, 1});
    statistics = image_statistics(img, 0);
    fprintf("Num kernels = %d\n", statistics(1));

    fprintf("Area: average = %.3fmm^2, median = %.3fmm^2\n", statistics(2), statistics(3))

    fprintf("MinorAxisLength: average = %.3fmm, median = %.3fmm\n", statistics(4), statistics(5))

    fprintf("MajorAxisLength: average = %.3fmm, median = %.3fmm\n\n", statistics(6), statistics(7))


    %Present statistics for RGB and HSV
    fprintf("r: average = %.3f, median = %.3f\n", statistics(8), statistics(9))
    fprintf("g: average = %.3f, median = %.3f\n", statistics(10), statistics(11))
    fprintf("b: average = %.3f, median = %.3f\n\n", statistics(12), statistics(13))
    fprintf("h: average = %.3f, median = %.3f\n", statistics(14), statistics(15))
    fprintf("s: average = %.3f, median = %.3f\n", statistics(16), statistics(17))
    fprintf("v: average = %.3f, median = %.3f\n\n\n", statistics(18), statistics(19))
end


%% 2.5 Answering the research questions
close all; clc;
clear all_statistics;
%column layout: 'Huvudskott', 'Gronskott', 'Gotala', 'Lanna', 'Multorp', 'Belinda', 'Fatima', 'Symphony'
categories_names = {'Huvudskott', 'Gronskott', 'Gotala', 'Lanna', 'Multorp', 'Belinda', 'Fatima', 'Symphony'};
statistics_names = {'Number of kernels', 'Average area', 'Median area', 'Average MinorAxisLength', 'Median MinorAxisLength', 'Average MajorAxisLength', 'Median MajorAxisLength', 'Average r', 'Median r', 'Average g', 'Median g', 'Average b', 'Median b', 'Average h', 'Median h', 'Average s', 'Median s','Average v', 'Median v'};

shoot_index_offset = 0;
farm_index_offset = shoot_index_offset + 2;
cultivation_index_offset = farm_index_offset + 3;
num_columns = 2 + 3 + 3;%2shots, 3farms, 3cultivations
num_statistics = 19;

all_statistics{num_statistics, num_columns} = [];


%for index = 1:number_of_images
for index = 1:number_of_images
    fprintf("Calculating statistics for image: %d\n", index);
    img = im2double(collection{index, 1});
    
    statistics = image_statistics(img, 0);
    tags = collection{index, 2};
    fprintf("\tstoring statistics\n");
    for k = 1:num_statistics
        column = shoot_index_offset + tags.shoot_index;
        all_statistics{k, column}{length(all_statistics{k, column}) + 1, 1} = statistics(k);
        
        column = farm_index_offset + tags.farm_index;
        all_statistics{k, column}{length(all_statistics{k, column}) + 1, 1} = statistics(k);
        
        column = cultivation_index_offset + tags.cultivation_index;
        all_statistics{k, column}{length(all_statistics{k, column}) + 1, 1} = statistics(k);
    end
    fprintf("\tdone with image: %d\n\n", index);
end


%Present one boxplot for each statistics
for i = 1:num_statistics
    data = [cell2mat(all_statistics{i, 1}); cell2mat(all_statistics{i, 2}); cell2mat(all_statistics{i, 3}); cell2mat(all_statistics{i, 4}); cell2mat(all_statistics{i, 5}); cell2mat(all_statistics{i, 6}); cell2mat(all_statistics{i, 7}); cell2mat(all_statistics{i, 8})];
    
    current_label_index = 1;
    label = [""];
    for k = 1:8
        num_components = length(all_statistics{i, k});
        for x = 1:num_components
            label(current_label_index, 1) = convertCharsToStrings(categories_names{k});
            current_label_index = current_label_index + 1;
        end
    end
    
    figure
    boxplot(data, label);
    ylabel(statistics_names{i});
end

