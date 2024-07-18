import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Apply Gaussian blur
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Adjust contrast
    image_enhanced = cv2.equalizeHist(image_blurred)
    return image_enhanced

def detect_vesicles(image):
    # Detect potential vesicle locations (using a simple threshold for example)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Find contours of detected vesicles
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vesicle_locations = np.array([cv2.boundingRect(contour)[:2] for contour in contours])
    return vesicle_locations

def apply_dbscan(data, eps=10, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    return labels

def plot_clusters(image, data, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.imshow(image, cmap='gray')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black used for noise.
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    plt.show()

def process_image_stack(file_path):
    image_stack = cv2.imreadmulti(file_path, [], cv2.IMREAD_GRAYSCALE)[1]
    
    preprocessed_stack = [preprocess_image(image) for image in image_stack]
    vesicle_locations_stack = [detect_vesicles(image) for image in preprocessed_stack]

    all_vesicle_locations = np.concatenate(vesicle_locations_stack)
    labels = apply_dbscan(all_vesicle_locations)

    for image, vesicle_locations in zip(preprocessed_stack, vesicle_locations_stack):
        plot_clusters(image, vesicle_locations, labels)
    
    return labels

# Example usage within the package
if __name__ == "__main__":
    file_path = 'path_to_your_tiff_stack.tiff'
    process_image_stack(file_path)
