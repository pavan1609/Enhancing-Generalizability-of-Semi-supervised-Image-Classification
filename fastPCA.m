function [reducedData, eigenvectors] = fastPCAMatlab(dataMatrix, numComponents)
  % This function performs PCA for dimensionality reduction using built-in MATLAB functions (optimized).

  % Validate input arguments
  if (~ismatrix(dataMatrix))
    error('Input dataMatrix must be a 2D matrix.');
  end
  if (numComponents < 1 || numComponents > size(dataMatrix, 2))
    error('Number of components must be between 1 and the number of columns in dataMatrix.');
  end

  % Center the data by subtracting the mean vector from each row (optimized using bsxfun)
  meanVector = mean(dataMatrix, 2);
  centeredData = bsxfun(@minus, dataMatrix, meanVector);

  % Compute the covariance matrix (optimized using cov)
  covarianceMatrix = cov(centeredData);

  % Find the top 'numComponents' eigenvectors and eigenvalues using eig
  [eigenvectors, eigenvalues] = eig(covarianceMatrix, 'vector');

  % Sort eigenvectors and eigenvalues in descending order of eigenvalues
  [eigenvalues, sortedIndices] = sort(eigenvalues, 'descend');
  eigenvectors = eigenvectors(:, sortedIndices);

  % Select the top 'numComponents' eigenvectors
  reducedData = centeredData * eigenvectors(:, 1:numComponents);
end
