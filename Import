function [faces] = importFaces(imagePath, numSubjects)
  % This function imports facial images and performs basic preprocessing.

  % Validate input arguments (optional)
  if (~ischar(imagePath) || ~exist(imagePath, 'dir'))
    error('imagePath must be a valid directory path.');
  end
  if (~isscalar(numSubjects) || numSubjects <= 0 || ~isreal(numSubjects))
    error('numSubjects must be a positive integer.');
  end

  % Initialize variables
  faces = struct([]);  % Pre-allocate empty struct array for faces
  subjectCount = 0;      % Counter for total number of faces

  % Loop through subject directories (s1, s2, ...)
  for subject = 1:numSubjects
    subjectPath = fullfile(imagePath, sprintf('s%d', subject));
    if ~exist(subjectPath, 'dir')
      warning(sprintf('Subject directory s%d not found. Skipping...', subject));
      continue;
    end

    % Get all image files (assuming .pgm extension)
    imageFiles = dir(fullfile(subjectPath, '*.pgm'));

    % Loop through each image file
    for imageIndex = 1:length(imageFiles)
      imageFile = imageFiles(imageIndex);

      % Read the image
      image = imread(fullfile(subjectPath, imageFile.name));

      % Convert to grayscale if needed (assuming RGB images)
      if (ndims(image) == 3)
        image = rgb2gray(image);
      end

      % Basic preprocessing (replace with your desired steps)
      processedImage = histeq(image);  % Histogram equalization for contrast enhancement

      % Feature extraction (replace with your preferred method)
      features = extractHOGFeatures(processedImage);  % Assuming extractHOGFeatures is available

      % Store information about the image
      subjectCount = subjectCount + 1;
      faces(subjectCount).subject = subject;
      faces(subjectCount).filename = imageFile.name;
      faces(subjectCount).image = image;
      faces(subjectCount).processedImage = processedImage;
      faces(subjectCount).features = features;
    end
  end
end
