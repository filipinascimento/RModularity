import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf8") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="RModularity",
    version="0.2.1",
    author="Filipi N. Silva",
    author_email="filipi@iu.edu",
    description="Python library to calculate Robustness modularity of networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[req for req in requirements if req[:2] != "# "],
    url="https://github.com/filipinascimento/RModularity",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)