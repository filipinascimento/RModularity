import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RModularity",
    version="0.2.0",
    author="Filipi N. Silva",
    author_email="filipi@iu.edu",
    description="Python library to calculate Robustness modularity of networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipinascimento/RModularity",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)