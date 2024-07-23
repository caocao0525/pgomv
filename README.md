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
If you use this code, please cite:
Seohyun Lee et al., "Image Processing Algorithm for Porphyromonas Gingivalis Outer Membrane Vesicle Transport in Periodental Pathogenesis", Submitted to IEEE NANOMED 2024.

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
ifw.process_and_save_tiff(input_file_path='path/to/input/image/stack',
    enhancement_method='scaling',
    alpha=10,  # Scaling factor for adjusting contrast. Higher values increase contrast.
    beta=10,  # Offset value for adjusting brightness. Higher values increase brightness.
    output_file_path='path/to/output/file/name',
    colormap='hot'
)

# Detect the vesicle locations using DBSCAN and convert the image to polar coordinates
ifw.detect_vesicles_and_convert_to_polar(file_path='path/to/your/enhanced/image/stack',
    dbscan_output_path='path/to/output/for/vesicle/detected',
    polar_output_path='path/to/output/for/polar/converted',
    eps=2,  # Adjust based on expected vesicle size
    min_samples=2,  # Adjust based on expected density of vesicle clusters
    threshold_value=128, # Adjust based on intensity threshold for vesicle detection
    line_thickness=5 # Thickness of the line used to label the detected vesicles
)

# Display the vesicle movements between consecutive frames, with diverging colormap
ifw.process_and_save_overlay(polar_image_path='path/to/your/polar/converted/image/stack',  
    initial_range=(start_frame, end_frame),   #  Range of frames to process, specified as a tuple of integers
    window_size=5,  # Size of the window (in pixels) to search for corresponding spots
    colormap='bwr'  # Diverging colormap, see Matplotlib diverging colormap
)


```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

