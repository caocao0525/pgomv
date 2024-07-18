import cv2
import numpy as np

def cartesian_to_polar(image, center):
    rows, cols = image.shape[:2]
    max_radius = int(np.sqrt(center[0]**2 + center[1]**2))
    polar_image = cv2.warpPolar(image, (cols, rows), center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_image
