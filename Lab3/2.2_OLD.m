   
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
    
    
    
    
    img_edges = img_vertical + img_horizontal;


    
   figure('name', 'Sobel edges'); 
   %imshow(img_edges); title(title_string);

     
figure('name', 'Binary image'); 
   
    img = img_bilateral;
    img = img > 0.65;
    
   %imshow(img); title(title_string);
   
   
   
   
   %% 
   
   
   
%index = randi(length(collection(:, 1)), 1);
%index = mod(index + 1, length(collection(:, 1))) + 1
fprintf("Current index:");
for index = 1:length(collection(:, 1))
    if mod(index - 1, 10) == 0
        fprintf("\n");
    end
    fprintf("%d\t", index);
    title_string = [num2str(index), ': ', farms{collection{index, 2}.farm_index},' ', cultivations{collection{index, 2}.cultivation_index},' ',shoots{collection{index, 2}.shoot_index}];

    figure('name', title_string);
    subplot(1, 2, 1); imshow(collection{index, 1})
    img = im2double(rgb2gray(collection{index, 1}));
    img = (img - min(img, [], 'all')) ./ ( max(img, [], 'all') - min(img, [], 'all') );

    %subplot(1, 4, 1); imshow(img); title(title_string); title("1) Normal");

    img_bilateral = img;%imbilatfilt(img, 0.1, 4, 'NeighborhoodSize', 3);
    img_bilateral = imfilter(img, fspecial('average', 3));
    %subplot(1, 4, 2);imshow(img_bilateral); title("2) Bilateral Filtering \sigma_r=0.1 \sigma_s=4 size=9x9");

    img = img_bilateral > 0.65;
    %subplot(1, 4, 3);imshow(img); title("3) Binary image. Threshold > 0.65");


    %img = medfilt2(img, [13, 13]);    
    %subplot(1, 4, 4);imshow(img); title("4) Median filter. 17x17 kernel");

    cc = bwconncomp(img);
    areas = regionprops(cc, 'Area');
    x = reshape(struct2array(areas), [], numel(fieldnames(areas)));

    remove_areas = find(x < 500); remove_areas = [remove_areas; find(x > 8000)];
    %figure("name", "FIlter"); imshow(img)

    img_remove = zeros(size(img));
    for i = 1:length(remove_areas)
        current_region = cc.PixelIdxList{remove_areas(i)}; 

        for k = 1:length(current_region)
            img_remove(current_region(k)) = 1;
        end
    end
    %figure("name", "Remove parts"); imshow(img_remove)

    img = img - img_remove;

    %figure("name",  "final"); 
    subplot(1, 2, 2); imshow(img)
end