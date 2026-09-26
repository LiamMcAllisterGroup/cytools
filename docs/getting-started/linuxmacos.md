---
title: Linux/macOS
---

# Linux/macOS

The installation of CYTools on Linux or macOS is primarily done using conda. A
conda environment allows us to package all of the dependencies and ensure that
everything will work properly. CYTools can also be installed with pip, but a few
optional dependencies (such as Normaliz) are not available that way.

## Requirements

- Linux: Almost any distribution, on a modern x86-64 or arm64 processor.
- macOS: A recent version of macOS on a Mac with an Apple silicon (M-series)
  processor. Intel-based Macs are not supported.
- (Optional but highly recommended) A conda installation.
  [Miniforge](https://conda-forge.org/) is recommended. The installation
  instructions below assume you are using conda, except for the pip
  installation.

## Installation instructions

=== "Easy installation"

    1. Install conda, if not already installed. Installation instructions can be
       found [here](https://conda-forge.org/download/).

    2. Run the following command in your terminal.

        ```bash
        curl -fsSL https://cy.tools/install.sh | bash
        ```

    3. Enjoy CYTools! :tada:

    You can take a look at exactly what the install script does by viewing it
    at [https://cy.tools/install.sh](https://cy.tools/install.sh) (that's the
    point of open-source software!). In short, it downloads an `environment.yml`
    file and then creates a conda environment from this file.

=== "Advanced installation"

    1. Install conda, if not already installed. Installation instructions can be
       found [here](https://conda-forge.org/download/).

    2. Download the source code of the latest stable release of CYTools from the
       following link, and extract the *zip* or *tar.gz* file into your desired
       location. Alternatively, clone the
       [repository](https://github.com/LiamMcAllisterGroup/cytools) with git.

        <div align="center" markdown>
        [![Download](../img/download.png){ width="200" }](https://github.com/LiamMcAllisterGroup/cytools/releases)
        </div>

    3. (Optional) Modify `environment-dev.yml` and any other files to fit your
       needs.

    4. Navigate to the root directory of CYTools in your terminal, and run
       `conda env create -f environment-dev.yml`. This will create a conda
       environment with CYTools, as well as its dependencies. CYTools is
       installed in editable mode, so any changes you make to its source code
       take effect without having to reinstall it.

=== "Desktop app"

    This method sets up a conda environment (offering to install Miniforge if
    conda is not found) and adds a CYTools launcher to your applications, which
    opens JupyterLab in the CYTools environment.

    1. Clone the [repository](https://github.com/LiamMcAllisterGroup/cytools),
       or download and extract the source code of the
       [latest release](https://github.com/LiamMcAllisterGroup/cytools/releases).

    2. Navigate to the root directory of CYTools in your terminal, and run the
       installer for your operating system.

        ```bash
        bash scripts/linux/install.sh  # On Linux
        bash scripts/macos/install.sh  # On macOS
        ```

    To uninstall the app, run `scripts/linux/uninstall.sh` or
    `scripts/macos/uninstall.sh` in the same way.

=== "pip"

    CYTools and its dependencies can be installed from PyPI with the following
    command. We recommend doing this in a
    [virtual environment](https://docs.python.org/3/library/venv.html).

    ```bash
    pip install cytools
    ```

    Note that the pip installation does not include Normaliz, which is needed
    for the computation of Hilbert bases. Use one of the conda installation
    methods if you need it. The [Mosek](advanced.md#mosek) optimizer and
    JupyterLab are also not included, but they can be installed with
    `pip install mosek jupyterlab`.

## Usage

The CYTools environment will be named either `cytools` if the easy installation
or the desktop app was used, or `cytools-dev` if the advanced installation was
followed. To enter this environment, type `conda activate cytools` or
`conda activate cytools-dev`, respectively. In this environment, `cytools` can
be imported in the same way as any other Python package such as NumPy, SciPy,
Matplotlib, etc. JupyterLab is installed by default, and it can be opened with
`jupyter lab`. To exit the conda environment, simply run `conda deactivate`.

## Updating

To update CYTools to the latest version, activate the `cytools` environment and
run `pip install --upgrade cytools`. If you followed the advanced installation,
update the source code instead (for example, with `git pull`).

## Removal

To remove the CYTools environment, run `conda env remove --name cytools` or
`conda env remove --name cytools-dev`, depending on which one you installed. If
you installed CYTools with pip, run `pip uninstall cytools` instead.

## Troubleshooting

Since there are many different platforms, it is possible that our installation
scripts don't always work. If this is the case, please let us know by emailing
us at [support@cy.tools](mailto:support@cy.tools) or by opening an issue on
[GitHub](https://github.com/LiamMcAllisterGroup/cytools/issues).
