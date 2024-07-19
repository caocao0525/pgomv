# Intraflow: Intracellular Movement Analysis and Visualization for pgOMVs

The Intraflow Image Processing Algorithm is a comprehensive workflow designed for the analysis of intracellular vesicle movements. This algorithm encompasses several key steps:

1. **Preprocessing and DBSCAN Clustering**
   * Enhances image quality and detects potential vesicle locations using DBSCAN clustering, a machine learning approach that effectively handles sparse data by identifying clusters of points and distinguishing noise.
  
2. **Polar Coordinate Conversion**
   * Converts the Cartesian coordinates of the image to polar coordinates to rearrange the images according to the cellular structure, with the cell center serving as the origin. This transformation facilitates specific types of image analysis that are centered around the cell's internal structure.

3. **Optical Flow Computation**
   * Computes and visualizes the optical flow between successive images to track the movement of vesicles over time.

4. **Visualization with Diverging Colormap**
   * Highlights high-intensity spots and visualizes the optical flow using a diverging colormap to distinguish the direction and magnitude of vesicle movements.

## Citation

## Installation

We suggest using `mamba` for managing your virtual environment due to its speed and efficiency. If you don't have `mamba` installed, you can install it using `conda` or by following the instructions on the [Mamba GitHub page](https://github.com/mamba-org/mamba).

1. **Install Mamba**:

    If you have `conda` installed:
    ```sh
    conda install mamba -n base -c conda-forge
    ```
2. Clone the repository:

    ```sh
    git clone https://github.com/caocao0525/pgomv.git
    cd pgomv
    ```

3. **Create and activate a mamba environment**:
    ```sh
    mamba env create -f environment.yml
    conda activate intraflow
    ```

4. Install the package:

    ```sh
    pip install -e .
    ```

## Usage

Here is an example of how to use the package:

```python
import cv2
import intraflow as ifw
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)

# Initial center (user designated or calculated)
initial_center = (image.shape[1] // 2, image.shape[0] // 2)

# Convert image to polar coordinates
polar_image = ifw.cartesian_to_polar(image, initial_center)

# Estimate center for a sequence of images
new_center = ifw.estimate_center(image, initial_center)

# Compute optical flow between two images
flow = ifw.compute_optical_flow(image, image)

# Draw optical flow on an image
flow_image = ifw.draw_optical_flow(flow, image)

# Highlight high intensity spots
highlighted_image = ifw.highlight_high_intensity(image, threshold=150)

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

