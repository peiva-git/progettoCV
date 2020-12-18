imagesNumber = 20;
squareSize = 30; % 30 mm square size

%%
% load all checkerboard images

I = imread(fullfile('images', 'Image1.tif'));
[rows, cols] = size(I);
images = zeros(rows, cols, imagesNumber);

for ii=1:imagesNumber
    imageFileName = strcat('Image', string(ii), '.tif');
    images(:, :, ii) = imread(fullfile('images', imageFileName));
end

%%
% load points using cade in lab1 with imageData structure array
% added board size for each image

clear imageData

for ii=1:imagesNumber
    imageFileName = fullfile('images', strcat('Image', string(ii), '.tif'));
    imageData(ii).image = imread(imageFileName);
    
    [imagePoints, boardSize] = detectCheckerboardPoints(imageData(ii).image);
    imageData(ii).XYpixels = imagePoints;
    imageData(ii).boardSize = boardSize;
end

%%
% establish correspondences, considering square size

for ii=1:imagesNumber
%    figure
%    imshow(imageData(ii).image, 'InitialMagnification', 300)
    
    clear Xmm Ymm
    for jj=1:length(imageData(ii).XYpixels)
        [row, col] = ind2sub([imageData(ii).boardSize(1) - 1, imageData(ii).boardSize(2) - 1], jj);
        Xmm = (col - 1) * squareSize;
        Ymm = (row - 1) * squareSize;
        
        imageData(ii).XYmm(jj, :) = [Xmm, Ymm];
%{        
        hndtxt = text(imageData(ii).XYpixels(jj, 1),...
            imageData(ii).XYpixels(jj, 2),...
            num2str([Xmm, Ymm]));
        set(hndtxt, 'fontsize', 14, 'color', [1 mod(row, 2) * mod(col, 2) 0]);
%}        
    end 
%    pause(1)
end

%%
