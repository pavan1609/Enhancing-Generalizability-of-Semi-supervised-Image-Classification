% Clear workspace and close figures
clc;
close all;

% Define path to your facial image folder
data_path = 'path/to/your/facial/images/';

% Load facial images and labels (subject for each image)
images = [];
labels = [];
file_list = dir(fullfile(data_path, '*.jpg'));  % Adjust for your image format
for i = 1:length(file_list)
  img = imread(fullfile(data_path, file_list(i).name));
  % Preprocess image (grayscale conversion, normalization)
  gray_img = rgb2gray(img);
  normalized_img = im2double(gray_img);
  % Resize image (optional for consistent size)
  resized_img = imresize(normalized_img, [size(normalized_img, 1) size(normalized_img, 2)]);
  images = [images; resized_img(:)];
  labels = [labels; i]; % Assuming subject corresponds to image index
end

% Calculate mean face
mean_face = mean(images, 1);

% Center data by subtracting mean face
centered_data = images - repmat(mean_face, size(images, 1), 1);

% Calculate covariance matrix
covariance_matrix = cov(centered_data');

% Eigenvalue decomposition (find top 'n' eigenvectors)
[eigenvectors, eigenvalues] = eig(covariance_matrix);
[sorted_eigenvalues, sorted_indices] = sort(diag(eigenvalues), 'descend');
n_eigenfaces = 100;  % Adjust the number of Eigenfaces
eigenfaces = eigenvectors(:, sorted_indices(1:n_eigenfaces));

% Project data onto Eigenface subspace
projected_data = centered_data * eigenfaces;

% Recognition function (replace with your testing logic)
function predicted_subject = recognize(new_image, mean_face, eigenfaces)
  % Preprocess new image
  new_gray_img = rgb2gray(new_image);
  new_normalized_img = im2double(new_gray_img);
  new_resized_img = imresize(new_normalized_img, [size(mean_face, 1) size(mean_face, 2)]);
  new_data = new_resized_img(:);
  centered_new_data = new_data - mean_face;
  projected_new_data = centered_new_data' * eigenfaces;
  
  % Calculate Euclidean distances to training data
  distances = sqrt(sum((projected_data - repmat(projected_new_data, size(projected_data, 1), 1)).^2, 2));
  
  % Identify subject with minimum distance
  [~, predicted_subject] = min(distances);
end

% Test on a new image (replace with your logic)
new_image_path = 'path/to/new/image.jpg';
new_image = imread(new_image_path);
predicted_subject = recognize(new_image, mean_face, eigenfaces);
disp(['Predicted Subject:', num2str(predicted_subject)]);

% Evaluation (replace with your evaluation method - K-Fold Cross-Validation)
% ... (Your evaluation logic here)
