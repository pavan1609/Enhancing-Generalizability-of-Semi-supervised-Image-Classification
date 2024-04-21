function [normalizedData, minValues, maxValues] = scaleData(dataMatrix, isTest, trainingMin, trainingMax)
  % This function performs min-max normalization of a data matrix.

  % Define target range for normalization (default: [-1, 1])
  targetMin = -1;
  targetMax = 1;

  % Check input arguments (optional)
  if (nargin < 2)
    isTest = false;  % Default: training data
  end
  if (~islogical(isTest))
    error('isTest argument must be logical (true or false).');
  end

  % Check training data parameters for testing (if applicable)
  if (isTest && nargin < 4)
    error('Training minimum and maximum values (trainingMin, trainingMax) are required for testing data.');
  end

  % Validate number of outputs (optional)
  if (nargout > 3)
    error('Maximum of three outputs are supported.');
  end

  % Get dimensions of the data matrix
  [numSamples, numFeatures] = size(dataMatrix);

  % Allocate memory for normalized data
  normalizedData = zeros(numSamples, numFeatures);

  % Perform normalization
  if (isTest)
    % Testing data normalization (using provided training min/max)
    for colIndex = 1:numFeatures
      if (trainingMax(colIndex) == trainingMin(colIndex))
        normalizedData(:, colIndex) = trainingMax(colIndex);
      else
        normalizedData(:, colIndex) = targetMin + ...
          (dataMatrix(:, colIndex) - trainingMin(colIndex)) / ...
          (trainingMax(colIndex) - trainingMin(colIndex)) * ...
          (targetMax - targetMin);
      end
    end
  else
    % Training data normalization (calculate min/max and store them)
    minValues = zeros(1, numFeatures);
    maxValues = zeros(1, numFeatures);
    for colIndex = 1:numFeatures
      minValues(colIndex) = min(dataMatrix(:, colIndex));
      maxValues(colIndex) = max(dataMatrix(:, colIndex));
      if (maxValues(colIndex) == minValues(colIndex))
        normalizedData(:, colIndex) = maxValues(colIndex);
      else
        normalizedData(:, colIndex) = targetMin + ...
          (dataMatrix(:, colIndex) - minValues(colIndex)) / ...
          (maxValues(colIndex) - minValues(colIndex)) * ...
          (targetMax - targetMin);
      end
    end
  end
end
