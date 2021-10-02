import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dbgz",
    version="0.3.4",
    author="Filipi N. Silva",
    author_email="filipi@iu.edu",
    description="Python library to load DBGZ files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipinascimento/dbgz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)