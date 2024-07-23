#!/usr/bin/env python
# coding: utf-8

# In[56]:


import os
import cv2
import numpy as np
import tifffile as tiff
from sklearn.cluster import DBSCAN
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from tqdm import tqdm


# In[47]:


# file_path = '../sample_data/crop1_HCAEC_Pg33277_4h.tif'  # Replace with the actual path to your TIFF file
# image_stack = tiff.imread(file_path)


# In[48]:


def apply_fourier_transform(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    return f_shift

def apply_inverse_fourier_transform(f_shift):
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def create_notch_filter(shape, centers, radius):
    rows, cols = shape
    mask = np.ones((rows, cols), np.uint8)
    
    x, y = np.ogrid[:rows, :cols]
    for center in centers:
        crow, ccol = center
        mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
        mask[mask_area] = 0
    
    return mask

def process_image(image, notch_centers, notch_radius):
    f_shift = apply_fourier_transform(image)
    
    notch_filter = create_notch_filter(image.shape, notch_centers, notch_radius)
    
    filtered_f_shift = f_shift * notch_filter
    
    filtered_image = apply_inverse_fourier_transform(filtered_f_shift)
    
    return filtered_image

def enhance_image(image, method='scaling', clip_limit=2.0, tile_grid_size=(8, 8), alpha=1.5, beta=50):
    if method == 'histogram_equalization':
        enhanced_image = cv2.equalizeHist(image.astype(np.uint8))
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image.astype(np.uint8))
    elif method == 'scaling':
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        raise ValueError("Invalid enhancement method. Choose 'histogram_equalization', 'clahe', or 'scaling'.")
    return enhanced_image

def normalize_intensity_across_stack(image_stack):
    # Calculate global mean and std deviation for the entire stack
    all_pixels = np.concatenate([image.flatten() for image in image_stack])
    global_mean = np.mean(all_pixels)
    global_std = np.std(all_pixels)
    
    # Normalize each image in the stack
    normalized_stack = []
    for image in image_stack:
        image = (image - np.mean(image)) / np.std(image)  # Standardize
        image = image * global_std + global_mean  # Scale to global stats
        image = np.clip(image, 0, 255)  # Clip to valid intensity range
        normalized_stack.append(image.astype(np.uint8))
    
    return np.array(normalized_stack)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    # For user-defined visualization
    image=image*brightness
    mean_img = np.mean(image)
    image = (image-mean_img)*contrast+mean_img
    image = np.clip(image,0,255)
    return image.astype(np.uint8)


# In[49]:


def process_and_save_tiff(input_path, enhancement_method, alpha, beta=10, output_path=None, colormap="hot"):
    # Load the image stack
    image_stack = tiff.imread(input_path)

    # Define the centers and radius for the notch filters (based on the interference pattern)
    notch_centers = [(image_stack[0].shape[0]//2, image_stack[0].shape[1]//2)]  # Adjust this based on the pattern
    notch_radius = 20  # Adjust this based on the pattern

    # Process and enhance each image in the stack
    processed_stack = []
    for i, original_image in enumerate(image_stack):
        filtered_image = process_image(original_image, notch_centers, notch_radius)
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_image = enhance_image(filtered_image, method=enhancement_method, alpha=alpha, beta=beta)
        processed_stack.append(enhanced_image)

    # Normalize intensity across the entire stack
    normalized_stack = normalize_intensity_across_stack(processed_stack)

    # Save the normalized stack to a new TIFF file
    if output_path:
        tiff.imwrite(output_path, normalized_stack)
        print("Your images are enhanced and saved at {}".format(output_path))

    # Optionally, display the images
    fig, axes=plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(image_stack[0], cmap=colormap)
    axes[0].set_title('First Raw Image')
    
    adjusted_image=adjust_brightness_contrast(image_stack[0], brightness=1.0, contrast=1.0)
    axes[1].imshow(adjusted_image, cmap=colormap)
    axes[1].set_title('First Adjusted Image')

    axes[2].imshow(normalized_stack[0], cmap=colormap)
    axes[2].set_title('First Normalized Enhanced Image')
    for ax in axes:
        ax.axis("off")

    plt.show()


# In[51]:


# # Example usage
# input_file_path = '../sample_data/crop1_HCAEC_Pg33277_4h.tif'
# output_file_path = '../../figs/normalized_image_stack.tif'
# enhancement_method = 'scaling'
# alpha = 10
# beta = 10

# # test
# process_and_save_tiff(input_file_path, enhancement_method, alpha, beta, output_file_path, colormap='hot')
# # test_passed


# In[52]:


# Global variable to store the selected center
selected_center = None

def select_center(event, x, y, flags, param):
    global selected_center
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_center = (x, y)
        print(f"Selected center: {selected_center}")

def load_image_stack(file_path):
    """Load the image stack from a TIFF file."""
    return tiff.imread(file_path)

def display_initial_image(image):
    """Display the initial image and allow the user to select the center."""
    cv2.namedWindow("Select Center")
    cv2.setMouseCallback("Select Center", select_center)
    
    while True:
        cv2.imshow("Select Center", image)
        if cv2.waitKey(1) & 0xFF == 27 or selected_center is not None:  # ESC key to exit
            break
    
    cv2.destroyAllWindows()
    return selected_center

def threshold_image(image, threshold_value=128):
    """Threshold the image to create a binary image."""
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def extract_coordinates(binary_image):
    """Extract coordinates of bright spots from the binary image."""
    coordinates = np.column_stack(np.where(binary_image > 0))
    return coordinates

def apply_dbscan(coordinates, eps=2, min_samples=2):
    """Apply DBSCAN clustering to the coordinates."""
    if len(coordinates) == 0:
        return np.array([]), np.array([])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = db.labels_
    # Filter out noise points
    core_coords = coordinates[labels != -1]
    core_labels = labels[labels != -1]
    return core_coords, core_labels

def draw_clusters(image, coordinates, labels, line_thickness=5):
    """Draw clusters on the original image with inverted colors and thicker lines."""
    inverted_image = cv2.bitwise_not(image)  # Invert the image colors
    output_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2RGB)
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels == label)
        xy = coordinates[class_member_mask]
        for point in xy:
            cv2.circle(output_image, (point[1], point[0]), line_thickness, (0, 0, 255), -1)  # Red points
    return output_image

def process_image_stack(file_path, eps=2, min_samples=2, threshold_value=128, output_path=None, line_thickness=5):
    """Process the entire image stack with DBSCAN and save the results."""
    # Load the image stack
    image_stack = load_image_stack(file_path)
    
    # Store DBSCAN coordinates for later use
    dbscan_coords_stack = []
    
    # Process each image in the stack
    result_stack = []
    for i, image in enumerate(image_stack):
        # Threshold the image to get a binary image
        binary_image = threshold_image(image, threshold_value)
        
        # Extract coordinates of potential vesicle positions
        coordinates = extract_coordinates(binary_image)
        
        # Apply DBSCAN to cluster the coordinates
        core_coords, core_labels = apply_dbscan(coordinates, eps, min_samples)
        
        # Store the DBSCAN-processed coordinates
        dbscan_coords_stack.append(core_coords)
        
        # Draw clusters on the original image
        result_image = draw_clusters(image, core_coords, core_labels, line_thickness)
        
        # Add the result image to the result stack
        result_stack.append(result_image)
    
    # Convert the result stack to a NumPy array
    result_stack = np.array(result_stack)
    
    # Save the result stack to a new TIFF file
    if output_path:
        tiff.imwrite(output_path, result_stack)
    
    return dbscan_coords_stack

def cartesian_to_polar(coordinates, center):
    """Convert Cartesian coordinates to polar coordinates."""
    x = coordinates[:, 1] - center[0]
    y = coordinates[:, 0] - center[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    polar_coordinates = np.column_stack((rho, phi))
    return polar_coordinates

def save_image_stack(file_path, image_stack):
    """Save the image stack to a TIFF file."""
    tiff.imwrite(file_path, image_stack)

def convert_coords_stack_to_polar(polar_image_path,dbscan_coords_stack, center, max_r, line_thickness=5):
    """Convert the DBSCAN coordinates in the image stack to polar coordinates."""
    polar_stack = []
    for coords in dbscan_coords_stack:
        polar_image = np.ones((int(max_r), 360, 3), dtype=np.uint8) * 255  # Create a white image with dimensions (max_r, 360, 3)
        if coords.size > 0:
            polar_coordinates = cartesian_to_polar(coords, center)
            rho = polar_coordinates[:, 0]
            phi = np.degrees(polar_coordinates[:, 1]) + 180  # Convert radians to degrees and shift to 0-360
            rho = np.clip(rho, 0, max_r - 1).astype(int)
            phi = np.clip(phi, 0, 359).astype(int)
            for r, p in zip(rho, phi):
                cv2.circle(polar_image, (p, r), line_thickness, (255, 0, 0), -1)  # Draw thicker red points
        polar_stack.append(polar_image)
    # Save the polar-converted image stack
    save_image_stack(polar_image_path, np.array(polar_stack))
    print(f"Polar converted image stack saved to {polar_image_path}")
    return np.array(polar_stack)


def detect_vesicles_and_convert_to_polar(input_file_path, dbscan_output_file_path, polar_output_file_path, eps=2, min_samples=2, threshold_value=128, line_thickness=5):
    # Ensure dbscan_output_file_path is a file path, not a directory
    if os.path.isdir(dbscan_output_file_path):
        dbscan_output_file_path = os.path.join(dbscan_output_file_path, "dbscan_detected_vesicle_stack.tif")
    
    # Ensure polar_output_file_path is a file path, not a directory
    if os.path.isdir(polar_output_file_path):
        polar_output_file_path = os.path.join(polar_output_file_path, "polar_converted_stack.tif")
    
    # Process the image stack with DBSCAN and save the results
    dbscan_coords_stack = process_image_stack(
        file_path=input_file_path, 
        eps=eps, 
        min_samples=min_samples, 
        threshold_value=threshold_value, 
        output_path=dbscan_output_file_path, 
        line_thickness=line_thickness
    )
    
    # Load the initial image for center selection
    initial_image = tiff.imread(input_file_path)[0]
    center = display_initial_image(initial_image)
    
    # Calculate the maximum radius for polar conversion
    max_r = np.sqrt(initial_image.shape[0]**2 + initial_image.shape[1]**2)
    
    # Convert coordinates to polar and save the polar stack
    polar_stack = convert_coords_stack_to_polar(polar_output_file_path, dbscan_coords_stack, center, max_r, line_thickness)
    
    return

    


# In[53]:


# # Example usage for DBSCAN processing
# file_path = '../../figs/normalized_image_stack.tif'  # Replace with the actual path to your TIFF file
# dbscan_output_path = '../../figs/dbscan_detected_vesicle_stack.tif'  # Replace with the desired output path for DBSCAN results
# polar_output_path = '../../figs/polar_converted_stack.tif'  # Replace with the desired output path for polar conversion results
# eps = 2  # Adjust based on expected vesicle size
# min_samples = 2  # Adjust based on expected density of vesicle clusters
# threshold_value = 128  # Adjust based on intensity threshold for vesicle detection
# line_thickness = 5  # Thickness of the line used to label the detected vesicles
# detect_vesicles_and_convert_to_polar(file_path, dbscan_output_path, polar_output_path, eps, min_samples, threshold_value, line_thickness)

# test passed


# In[71]:


def extract_spot_locations(image, threshold_value=128):
    """Extract coordinates of spots from the binary image."""
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    coordinates = np.column_stack(np.where(binary_image > 0))
    return coordinates

def find_moving_spots_with_vectors(coords1, coords2, window_size=5):
    """Find spots in coords1 that have corresponding spots in coords2 within the window size and calculate their movement vectors."""
    moving_spots = []
    vectors = []
    for (y1, x1) in coords1[:, :2]:  # Ensure only y and x coordinates are used
        # Limit the search to a window around (y1, x1)
        window_coords = coords2[(abs(coords2[:, 0] - y1) <= window_size) & (abs(coords2[:, 1] - x1) <= window_size)][:, :2]
        for (y2, x2) in window_coords:
            if abs(y1 - y2) <= window_size and abs(x1 - x2) <= window_size:
                moving_spots.append((y1, x1))
                vectors.append((y2 - y1, x2 - x1))
                break
    return np.array(moving_spots), np.array(vectors)

def calculate_angles_and_color(moving_spots, vectors, colormap='bwr'):
    """Calculate the angles between the movement vectors and the vectors from the origin and apply colormap."""
    origin = (180, 0)
    colors = []
    for i, (spot, vector) in enumerate(zip(moving_spots, vectors)):
        vector1 = np.array([spot[0] - origin[0], spot[1] - origin[1]])
        vector2 = vector
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            angle = 90  # Default to 90 degrees if either vector is zero to indicate no movement
        else:
            dot_product = np.dot(vector1, vector2)
            # Ensure the dot product result is within the valid range for arccos
            dot_product = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
        
        # Normalize the angle to the range [0, 180]
        angle = min(angle, 180 - angle)
        
        # Determine the color based on the angle
        if angle < 90:
            color = angle / 90  # Scale to [0, 1]
        else:
            color = (angle - 90) / 90  # Scale to [0, 1]
        
        colors.append(color)
    
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(colormap)
    colored_spots = [cmap(norm(color)) for color in colors]
    
    return colored_spots


def process_and_save_overlay(polar_image_path, initial_range=(30, 40), window_size=5, colormap='bwr'):
    # Load the polar-converted image stack
    polar_stack = load_image_stack(polar_image_path)

    # Create subplots
    num_images = initial_range[1] - initial_range[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 5))  # Adjust figsize for better spacing

    # Process each pair of consecutive images in the specified range of the stack
    for i, ax in zip(tqdm(range(initial_range[0], initial_range[1]), desc="Processing images"), axes):
        first_image = polar_stack[i]
        second_image = polar_stack[i + 1]

        # Invert the images to make the background white and spots black
        first_image_inverted = cv2.bitwise_not(first_image)
        second_image_inverted = cv2.bitwise_not(second_image)

        # Extract spot locations from both images
        coords1 = extract_spot_locations(first_image_inverted)
        coords2 = extract_spot_locations(second_image_inverted)

        # Find moving spots and their movement vectors
        moving_spots, vectors = find_moving_spots_with_vectors(coords1, coords2, window_size)

        # Calculate angles and determine colors
        colored_spots = calculate_angles_and_color(moving_spots, vectors, colormap)

        # Create an image to display
        overlay_image = np.ones((polar_stack.shape[1], polar_stack.shape[2], 3), dtype=np.float32)  # Initialize with white background
        for (spot, color) in zip(moving_spots, colored_spots):
            cv2.circle(overlay_image, (spot[1], spot[0]), 10, color, -1)

        # Display the overlay image in the subplot
        ax.imshow(overlay_image)
        ax.set_title(f'Vesicle Movement {i+1} to {i+2}')
        ax.axis('off')

    # Adjust layout to reduce space between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between plots
    plt.show()


# In[72]:


# # Example usage
# polar_image_path = '../../figs/polar_converted_stack.tif'
# overlay_output_path = '../../figs/vesicle_movement_colored.tif'
# process_and_save_overlay(polar_image_path, initial_range=(30, 32), window_size=5, colormap='bwr')
# # test passed


# In[ ]:




