import cv2
import numpy as np

def cartesian_to_polar(image, center):
    # Calculate maximum radius from the center to the furthest corner
    height, width = image.shape
    max_radius = np.hypot(center[0], center[1])
    
    # Ensure that the max_radius covers the farthest corner in a rectangular image
    max_radius = max(np.hypot(center[0], center[1]), np.hypot(center[0], height - center[1]), 
                     np.hypot(width - center[0], center[1]), np.hypot(width - center[0], height - center[1]))

    # Convert Cartesian to Polar coordinates
    polar_image = cv2.linearPolar(image, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_image

