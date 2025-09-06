"""
Setup script for the Coreference Resolution library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        content = fh.read()
        # Add note about additional requirements
        additional_note = "\n\n**Note**: This package requires the CorefUD scorer tools for evaluation, which must be installed separately. See the README for installation instructions.\n"
        return content + additional_note

# Read requirements
def read_requirements():
    requirements_path = os.path.join("config", "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle git dependencies
                if line.startswith('unsloth'):
                    # Convert to proper pip format for setup.py
                    requirements.append('unsloth @ git+https://github.com/unslothai/unsloth.git')
                else:
                    requirements.append(line)
        return requirements

# Read version from __init__.py
def read_version():
    with open("src/__init__.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split('"')[1]
    return "0.1.0"

setup(
    name="coref-inst",
    version=read_version(),
    author="Tuğba Pamay Arslan, Emircan Erol, Gülşen Eryiğit",
    author_email="tpamay@itu.edu.tr, erole20@itu.edu.tr",
    description="Coreference Resolution with Large Language Models - Research Implementation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/itunlplab/coref-inst",
    project_urls={
        "Bug Tracker": "https://github.com/itunlplab/coref-inst/issues",
        "Documentation": "https://github.com/itunlplab/coref-inst#readme",
        "Source Code": "https://github.com/itunlplab/coref-inst",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "pre-commit>=2.15",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
            "matplotlib>=3.5",
        ],
        "evaluation": [
            "scikit-learn>=1.0",
            "pandas>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "coref-train=src.training.train_unified:main",
            "coref-infer=src.inference.infer:main",
            "coref-process-data=src.data_processing.format_dataset.dataset:main",
            "coref-map-predictions=src.evaluation.mapper.map_pred:main",
        ],
    },
    package_data={
        "src": ["*.py"],
        "config": ["instructions/*.txt", "requirements.txt"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "coreference resolution",
        "natural language processing",
        "large language models",
        "transformers",
        "fine-tuning",
        "nlp",
        "ai",
        "machine learning",
    ],
)
