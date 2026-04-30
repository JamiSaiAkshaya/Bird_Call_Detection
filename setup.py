"""
Setup script for AI-Powered Bird Call Detection project
Professional package configuration for easy installation and distribution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="bird-call-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered bird call detection system for wildlife conservation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/bird-call-detection",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/bird-call-detection/issues",
        "Documentation": "https://github.com/your-username/bird-call-detection/wiki",
        "Source Code": "https://github.com/your-username/bird-call-detection",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.23.1",
            "ipywidgets>=8.0.0",
        ],
        "deployment": [
            "gunicorn>=20.1.0",
            "uvicorn>=0.20.0",
            "fastapi>=0.95.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bird-detect=src.inference.realtime_detector:main",
            "bird-train=src.training.trainer:main",
            "bird-api=src.api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    data_files=[
        ("config", ["config/config.yaml"]),
    ],
    zip_safe=False,
    keywords=[
        "bird call detection",
        "wildlife conservation", 
        "audio classification",
        "deep learning",
        "pytorch",
        "efficientnet",
        "endangered species",
        "bioacoustics",
        "conservation technology",
    ],
)