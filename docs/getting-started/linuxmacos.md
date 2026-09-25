---
title: Linux/macOS
---

# Linux/macOS

The installation of CYTools on Linux or macOS is primarily done using conda. A
conda environment allows us to package all of the dependencies and ensure that
everything will work properly. CYTools can also be installed with pip, but a few
optional dependencies (such as Normaliz) are not available that way.

## Requirements

- Almost any Linux distribution or any recent version of macOS (>=10.13).
- Linux: A modern x86-64 or arm64 processor.
- macOS: Any modern Apple computer with an Apple silicon (M-series) processor.
  Intel-based Macs are not supported.
- (Optional but highly recommended) A conda installation.
  [Miniforge](https://conda-forge.org/) is recommended. The installation
  instructions below assume you are using conda.

## Installation instructions

=== "Easy installation"

    1. Install conda, if not already installed. Installation instructions can be
       found [here](https://conda-forge.org/download/).

    2. Run the following command on your terminal.

        ```bash
        curl https://cy.tools/install.sh | bash
        ```

    3. Enjoy CYTools! :tada:

=== "Advanced installation"

    1. Install conda, if not already installed. Installation instructions can be
       found [here](https://conda-forge.org/download/).

    2. Download the source code of the latest stable release of CYTools from the
       following link, and extract the *zip* or *tar.gz* file into your desired
       location.

        <div align="center" markdown>
        [![Download](../img/download.png){ width="200" }](https://github.com/LiamMcAllisterGroup/cytools/releases)
        </div>

    3. (Optional) Modify `environment-dev.yml` and any other files to fit your
       needs.

    4. Navigate to the root directory of CYTools in your terminal, and run
       `conda env create -f environment-dev.yml`. This will create a conda
       environment with CYTools, as well as its dependencies.

=== "pip"

    CYTools and its dependencies can be installed from PyPI with the following
    command.

    ```bash
    pip install cytools
    ```

    Note that the pip installation does not include Normaliz, which is needed
    for a few functions such as the computation of Hilbert bases. Use one of the
    conda installation methods if you need it.

You can take a look at exactly what the install script does by viewing it at
[https://cy.tools/install.sh](https://cy.tools/install.sh) (that's the point of
open-source software!). In short, it downloads an `environment.yml` file and then
creates a conda environment from this file.

## Usage

The CYTools environment will be named either `cytools` if the standard install
was performed or `cytools-dev` if the advanced installation was followed. To
enter this environment, type `conda activate cytools` or
`conda activate cytools-dev`, respectively. In this environment, `cytools` can
be imported in the same way as any other Python package such as NumPy, SciPy,
Matplotlib, etc. By default, JupyterLab is installed, which can simply be opened
with `jupyter lab`. To exit the conda environment, simply call
`conda deactivate`.

## Removal

To remove the CYTools environment, run `conda env remove --name cytools` or
`conda env remove --name cytools-dev`, depending on which version (normal or
dev) you installed. If you installed CYTools with pip, run
`pip uninstall cytools` instead.

## Troubleshooting

Since there are many different platforms, it is possible that our installation
scripts don't always work. If this is the case, please let us know by emailing
us at [support@cy.tools](mailto:support@cy.tools) or by opening an issue on
[GitHub](https://github.com/LiamMcAllisterGroup/cytools/issues).
