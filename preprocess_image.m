% Clear workspace and close figures
clc;
close all;

% Function to read and preprocess an image
function [gray_face, face_detected] = preprocess_image(image_path)
  % Read image
  try
    image = imread(image_path);
  catch ME
    warning('Error reading image: %s', ME.message);
    gray_face = [];
    face_detected = false;
    return;
  end

  % Convert to grayscale
  gray_image = rgb2gray(image);

  % Optional face detection using Viola-Jones algorithm
  face_detector = vision.CascadeObjectDetector('FrontalFaceLBPH');
  bboxes = step(face_detector, gray_image);

  % Check if a face is detected
  face_detected = ~isempty(bboxes);

  if face_detected
    % Extract the first detected face (adjust if needed)
    face_bbox = bboxes(1, :);
    gray_face = imcrop(gray_image, face_bbox);
  else
    gray_face = [];
  end
end

% Function for Eigenface-based prediction
function predicted_label = predict_using_eigenfaces(gray_face, model)
  % Resize face to match training size
  resized_face = imresize(gray_face, size(model.eigenfaces, 1:2));

  % Center data by subtracting mean face
  centered_face = resized_face(:) - model.mean_face(:);

  % Project face onto Eigenface subspace
  projected_face = model.eigenfaces' * centered_face;

  % Calculate Euclidean distances to training data projections
  distances = sqrt(sum((projected_face - model.projected_training_data).^2, 2));

  % Find nearest neighbor (consider k-Nearest Neighbors for improvement)
  [~, predicted_label] = min(distances);
  predicted_label = model.labels(predicted_label);
end

% Load or train the Eigenface model (replace with your implementation)
% This section is a placeholder, replace with your actual training logic
% for Eigenfaces using PCA and data preparation
model.eigenfaces = ...;  % Replace with your trained Eigenfaces
model.mean_face = ...;   % Replace with your calculated mean face
model.projected_training_data = ...;  % Replace with projected training data
model.labels = ...;      % Replace with training data labels

% Test on a pre-loaded image (replace with your path)
image_path = 'path/to/your/image.jpg';
[gray_face, face_detected] = preprocess_image(image_path);

if face_detected
  % Predict label using Eigenfaces
  predicted_label = predict_using_eigenfaces(gray_face, model);
  disp(['Predicted Label:', num2str(predicted_label)]);

  % Display image with predicted label
  figure;
  imshow(gray_face);
  title(['Predicted Label:', num2str(predicted_label)]);
else
  disp('No face detected in the image.');
end

% Alternatively, test on a set of images in a folder (adjust folder path)
image_folder = 'path/to/your/image/folder/';
image_files = dir(fullfile(image_folder, '*.jpg'));

for i = 1:length(image_files)
  image_path = fullfile(image_folder, image_files(i).name);
  [gray_face, face_detected] = preprocess_image(image_path);

  if face_detected
    predicted_label = predict_using_eigenfaces(gray_face, model);
    disp(['Image:', image_files(i).name, ', Predicted Label:', num2str(predicted_label)]);
  else
    disp(['No face detected in image:', image_files(i).name]);
  end
end
