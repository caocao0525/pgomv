import cv2
import numpy as np

def estimate_center(image, prev_center, window_size=50):
    x, y = prev_center
    half_window = window_size // 2
    region = image[max(0, y-half_window):min(image.shape[0], y+half_window),
                   max(0, x-half_window):min(image.shape[1], x+half_window)]
    
    # Find the point of highest intensity within the window
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(region)
    new_center = (x + max_loc[0] - half_window, y + max_loc[1] - half_window)
    
    return new_center

def highlight_high_intensity(image, threshold):
    mask = image > threshold
    highlighted = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    return highlighted
