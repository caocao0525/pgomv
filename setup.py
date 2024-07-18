from setuptools import setup, find_packages

setup(
    name='intraflow',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    author='Seohyun Lee',
    author_email='seohyun.lee@iii.u-tokyo.ac.jp',
    description='A package for intracellular movement pattern of pg OMV',
    url='https://github.com/caocao0525/pgomv',  # Replace with your GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
