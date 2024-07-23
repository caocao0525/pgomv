# Intraflow: Intracellular Movement Analysis and Visualization for P. gingivalis OMVs

The Intraflow Image Processing Algorithm is a comprehensive workflow designed to analyze intracellular vesicle movements. This algorithm encompasses several key steps:

1. **Preprocessing using Fourier Transform and DBSCAN**
   * Enhances image quality via Fourier Transform filter and detects potential vesicle locations using DBSCAN, a machine-learning approach that effectively handles sparse data by identifying clusters of points and distinguishing noise.
  
2. **Polar Coordinate Conversion**
   * Converts the Cartesian coordinates of the image to polar coordinates to rearrange the images according to the cellular structure, with the cell center serving as the origin. This transformation facilitates specific image analysis types centered around the cell's internal structure.

3. **Inward/Outward Movement Determination**
   * Calculates the angles between two vectors at the vesicle's position: one vector points towards the cell center, and the other points to the vesicle's position in the subsequent time stamp. This calculation helps determine the direction of vesicle movement relative to the cell center.

4. **Visualization with Diverging Colormap**
   * Inward movements are colored red, while outward movements are colored blue.

## Citation
Under review: Conference name

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
import intraflow as ifw

# Process and save the initial image stack (Gaussian filter, Normalization, Inverse Fourier Transform)
process_and_save_tiff(input_file_path='path/to/input/image/stack', enhancement_method='scaling', alpha=10, beta=10, output_file_path='path/to/output/file/name, colormap='hot')

# alpha: Scaling factor for adjusting contrast. Higher values increase contrast.
# beta: Offset value for adjusting brightness. Higher values increase brightness.
```
```python
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

