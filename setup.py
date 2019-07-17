import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-your-username",
    version="0.0.1",
    author="Lana Andreasyan",
    author_email="lana.andreasyan@gmail.com",
    description="Implementation of Locally Linear Regression of graph linked data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/landreasyan/LLR.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)