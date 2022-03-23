import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cytools",
    version="0.4.1",
    author="Liam McAllister Group",
    author_email="",
    description="A software package for analyzing Calabi-Yau hypersurfaces in toric varieties.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LiamMcAllisterGroup/cytools",
    packages=setuptools.find_packages(),
    license="GNU General Public License (GPL)",
    python_requires='>=3.6',
    install_requires=[], #["numpy", "scipy", "cvxopt", "flint-py", "pplpy", "ortools"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ]
)
