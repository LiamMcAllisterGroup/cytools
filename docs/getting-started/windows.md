---
title: Windows
---

# Windows

CYTools is installed on Windows through the Windows Subsystem for Linux (WSL).
Apart from this, the installation is identical to that for a Linux system.

## Requirements

- A recent version of Windows 10 or 11.
- A modern x86-64 or arm64 processor with hardware virtualization enabled
  (required by WSL 2).
- A WSL installation.

## Installation instructions

1. Install the Windows Subsystem for Linux (WSL), if not already installed.
   Installation instructions can be found
   [here](https://learn.microsoft.com/en-us/windows/wsl/install).

2. Open the WSL terminal and ensure that conda is installed. Installation
   instructions can be found
   [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows-subsystem-for-linux-wsl).
   Then source your `.bashrc` (or simply reopen the terminal).

3. Follow the [Linux/macOS](linuxmacos.md) installation instructions in this
   terminal. Namely, run the following command.

    ```bash
    curl -fsSL https://cy.tools/install.sh | bash
    ```

4. Enjoy CYTools! :tada:

You can take a look at exactly what the install script does by viewing it at
[https://cy.tools/install.sh](https://cy.tools/install.sh) (that's the point of
open-source software!). In short, it downloads an `environment.yml` file and then
creates a conda environment from this file.

## Usage

The CYTools environment will be named `cytools`. To enter this environment, type
`conda activate cytools` in the WSL terminal. In this environment, `cytools` can
be imported in the same way as any other Python package such as NumPy, SciPy,
Matplotlib, etc. JupyterLab is installed by default, and it can be opened with
`jupyter lab` (the webpage might not open automatically, in which case you can
copy the URL printed in the terminal into your browser). To exit the conda
environment, simply run `conda deactivate`.

## Updating

To update CYTools to the latest version, open the WSL terminal, activate the
`cytools` environment, and run `pip install --upgrade cytools`.

## Removal

To remove the CYTools environment, first open the WSL terminal. In this
terminal, run `conda env remove --name cytools`.

## Troubleshooting

Since most of the testing of CYTools is done on Linux, it is possible that our
installation scripts don't always work on WSL. If this is the case, please let
us know by emailing us at [support@cy.tools](mailto:support@cy.tools) or by
opening an issue on [GitHub](https://github.com/LiamMcAllisterGroup/cytools/issues).
