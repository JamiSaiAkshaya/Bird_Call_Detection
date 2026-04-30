"""
Setup script for AI-Powered Bird Call Detection — ML Extension v2
"""

from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="bird-call-detection",
    version="2.0.0",
    author="JamiSaiAkshaya",
    author_email="230701121@rajalakshmi.edu.in",
    description="AI-powered bird call detection system for wildlife conservation — ML Extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JamiSaiAkshaya/Bird_Call_Detection",
    packages=find_packages(where="."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    # Keep install_requires minimal — user installs torch separately
    # and requirements.txt handles everything else.
    install_requires=[
        "numpy>=1.26.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "ml": [
            "timm>=0.9.0",
            "onnx>=1.15.0",
            "onnxruntime>=1.17.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
        ],
    },
    include_package_data=True,
    package_data={"": ["*.yaml", "*.yml", "*.json", "*.md"]},
    zip_safe=False,
)
