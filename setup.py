import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="fear_n_greed",
    version="0.0.1",
    author="JAE-HYEONG LEE",
    author_email="jhyj121000@gmail.com",
    description="To get FEAR and GREED index with price and volume data.",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaealways/fear-and-greed.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)