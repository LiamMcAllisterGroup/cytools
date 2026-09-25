---
title: Windows
---

# Windows

The installation of CYTools on Windows is exclusively done using the Windows
Subsystem for Linux (WSL). Except for this, the installation is identical to
that for a Linux/macOS system.

## Requirements

- A recent version of Windows 10 or 11.
- Any modern x86-64 processor with hardware virtualization enabled. Other
  architectures might work with emulation, but there could be problems.
- A WSL installation.

## Installation instructions

1. Install Windows Subsystem for Linux (WSL), if not already installed.
   Installation instructions can be found
   [here](https://learn.microsoft.com/en-us/windows/wsl/install).

2. Open the WSL terminal and ensure that conda is installed. Installation
   instructions can be found
   [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows-subsystem-for-linux-wsl).
   Then source your `.bashrc` (or simply reopen the terminal).

3. Follow the [Linux/macOS](linuxmacos.md) installation in this terminal.
   Namely, run the following command.

    ```bash
    curl https://cy.tools/install.sh | bash
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
Matplotlib, etc. By default, JupyterLab is installed, which can simply be opened
with `jupyter lab` (note: the webpage might not be automatically opened; simply
enter the URL in your browser). To exit the conda environment, simply call
`conda deactivate`.

## Removal

To remove the CYTools environment, first open the WSL terminal. In this
terminal, run `conda env remove --name cytools`.

## Troubleshooting

Since most of the testing for CYTools is done on Linux, it is possible that our
installation scripts don't always work on WSL. If this is the case, please let
us know by emailing us at [support@cy.tools](mailto:support@cy.tools) or by
opening an issue on [GitHub](https://github.com/LiamMcAllisterGroup/cytools/issues).
