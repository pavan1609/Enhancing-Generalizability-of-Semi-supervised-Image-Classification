from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2  # Assuming OpenCV is installed for image processing

# Define function to extract features (replace with your preferred method)
def extract_features(image):
  # Preprocess the image (grayscale conversion, normalization, etc.)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Extract features (e.g., using HOG)
  features = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray)
  return features.flatten()

# Load your facial images and labels (subject for each image)
images = []
labels = []
# ... (Your image loading and labeling logic here)

# Extract features from images
features = [extract_features(img) for img in images]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # Adjust the number of components as needed
features_reduced = pca.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear')  # Choose the appropriate kernel (e.g., linear, rbf)
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use the trained model for prediction on new images
new_image = cv2.imread("path/to/new/image.jpg")
new_feature = extract_features(new_image)
new_feature_reduced = pca.transform([new_feature])
predicted_label = svm.predict(new_feature_reduced)[0]
print("Predicted subject:", predicted_label)
