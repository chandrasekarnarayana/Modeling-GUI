from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="modeling_gui",  # Package name
    version="0.1.1",  # Initial version
    author="Chandrasekar SUBRAMANI NARAYANA",
    author_email="chandrasekarnarayana@gmail.com",
    description="A PyQt5-based GUI for running statistical and machine learning models with customizable parameters and visualizations.",
    long_description=long_description,  # Detailed description from README
    long_description_content_type="text/markdown",  # README content type
    url="https://github.com/your_username/modeling_gui",  # Replace with your GitHub repo URL
    packages=find_packages(),  # Automatically find all packages in this directory
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    install_requires=[
        "pyqt5>=5.15.4",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "pandas>=1.3.3",
        "statsmodels>=0.12.2",
        "scikit-learn>=0.24.2",
        "scipy>=1.7.1",
        "graphviz>=0.16",
        "numpy>=1.21.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Minimum Python version is now 3.9
    entry_points={
        'console_scripts': [
            'run_modeling_gui=modeling_gui.main:main',  # Entry point for running the GUI from CLI
        ],
    },
)
