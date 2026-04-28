from setuptools import setup, find_packages

setup(
    name="hallscan",
    version="0.1.0",
    author="Nikhil Upadhyay",
    author_email="nikhil25000@gmail.com",
    description="Hallucination detection for language models using internal activation analysis",
    long_description="HallScan detects hallucination patterns in language models by analyzing internal attention maps and hidden states. Based on the paper: Hallucination Fingerprints (Upadhyay, 2026).",
    long_description_content_type="text/markdown",
    url="https://github.com/TrazeMaG/hallucination-fingerprints",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)