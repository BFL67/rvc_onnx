from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="rvc-onnx",
    version="1.0.0",
    author="RVC ONNX Contributors",
    author_email="",
    description="ONNX-based implementation for Retrieval-based Voice Conversion (RVC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BFL67/RVC_Onnx",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rvc-onnx-convert=rvc_onnx.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

