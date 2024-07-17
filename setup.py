from setuptools import setup, find_packages

setup(
    name='my_image_processing_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for image processing including polar conversion and optical flow.',
    url='https://github.com/yourusername/my_image_processing_package',  # Replace with your GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
