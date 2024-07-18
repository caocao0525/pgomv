# Cell-wise Intracellular Movement Analysis and Visualization for pgOMVs

A Python package for image processing, including Cartesian to Polar conversion, optical flow computation, and highlighting high-intensity spots in images.

## Features

- **Cartesian to Polar Conversion**: Convert images from Cartesian coordinates to polar coordinates.
- **Optical Flow Computation**: Compute and visualize the optical flow between two images.
- **Center Estimation**: Estimate the center of an image based on intensity.
- **Highlight High-Intensity Spots**: Highlight high-intensity spots in images using a diverging colormap.

## Citation

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/my_image_processing_package.git
    cd my_image_processing_package
    ```

2. Create and activate a conda environment:

    ```sh
    conda env create -f environment.yml
    conda activate my_image_processing_env
    ```

3. Install the package:

    ```sh
    pip install -e .
    ```

## Usage

Here is an example of how to use the package:

```python
import cv2
import my_image_processing_package as mip
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)

# Initial center (user designated or calculated)
initial_center = (image.shape[1] // 2, image.shape[0] // 2)

# Convert image to polar coordinates
polar_image = mip.cartesian_to_polar(image, initial_center)

# Estimate center for a sequence of images
new_center = mip.estimate_center(image, initial_center)

# Compute optical flow between two images
flow = mip.compute_optical_flow(image, image)

# Draw optical flow on an image
flow_image = mip.draw_optical_flow(flow, image)

# Highlight high intensity spots
highlighted_image = mip.highlight_high_intensity(image, threshold=150)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Polar Image')
plt.imshow(polar_image, cmap='gray')
plt.axis('off')

plt.show()

```

## Running Tests

To run tests, use pytest:

```sh
pytest tests
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

