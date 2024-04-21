% Define paths (replace with your actual paths)
dataPath = 'path/to/your/dataset';
trainingImagesPath = fullfile(dataPath, 'training');
testingImagePath = fullfile(dataPath, 'testing', 'test_image.jpg');

% Read training images and labels
trainingFiles = dir(fullfile(trainingImagesPath, '*.jpg'));
numImages = numel(trainingFiles);
trainingFaces = zeros(size(imread(fullfile(trainingImagesPath, trainingFiles(1).name))), [numImages 1]); % Pre-allocate for efficiency
trainingLabels = strings(numImages, 1);

for i = 1:numImages
  imagePath = fullfile(trainingImagesPath, trainingFiles(i).name);
  [~, name, ~] = fileparts(imagePath); % Extract name from filename (assuming it represents the label)
  trainingFaces(:,:,i) = imresize(im2double(rgb2gray(imread(imagePath))), [100 100]); % Resize and normalize
  trainingLabels(i) = name;
end

% Apply PCA for feature extraction (adjust number of components)
[eigenfaces, ~, variances] = pca(trainingFaces(:,:), 100); % Extract 100 eigenfaces

% Project training faces onto the eigenspace
projectedFaces = eigenfaces' * trainingFaces;

% Function to recognize a face in a new image
function [recognizedLabel, distance] = recognizeFace(testImage, eigenfaces, projectedFaces, trainingLabels)
  % Preprocess test image
  testImage = imresize(im2double(rgb2gray(testImage)), [100 100]);
  
  % Project test image onto the eigenspace
  projectedTest = eigenfaces' * testImage;
  
  % Implement KNN classification (replace with your preferred method)
  distances = sqrt(sum(bsxfun(@minus, projectedTest, projectedFaces).^2, 2)); % Calculate Euclidean distances
  [minDistance, minIndex] = min(distances);
  recognizedLabel = trainingLabels(minIndex);
  distance = minDistance;
end

% Load testing image
testImage = imresize(im2double(rgb2gray(imread(testingImagePath))), [100 100]);

% Face recognition using the function
[recognizedLabel, distance] = recognizeFace(testImage, eigenfaces, projectedFaces, trainingLabels);

% Display results (optional)
figure(1);
subplot(2,1,1);
imshow(testImage);
title('Test Image');
subplot(2,1,2);
imshow(trainingFaces(:,:,find(trainingLabels == recognizedLabel, 1))); % Display matching training image (modify for better visualization)
title(sprintf('Recognized: %s (Distance: %.2f)', recognizedLabel, distance));

disp(sprintf('Recognized face belongs to: %s with distance: %.2f', recognizedLabel, distance));
