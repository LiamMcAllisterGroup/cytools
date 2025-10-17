---
id: windows
title: Windows
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The installation of CYTools on Windows is exclusively done using the Windows Subsystem for Linux (WSL). Except for this, the installation is identical to that for a Linux/macOS system.

utilizes . Other than WSL, this install is identical

## Requirements

- A recent version of Windows 10 or 11.
- Any modern x86-64 processor with hardware virtualization enabled. Other architectures might work with emulation, but there could be problems.
- A WSL installation.

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

1. Install Windows Subsystem for Linux (WSL), if not already installed. Installation instructions can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install).

2. Open the WSL terminal and ensure that conda is installed. Installation instruction can be found [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows-subsystem-for-linux-wsl). Source the bashrc (can also be sourced by reopening the terminal).

3. Follow the Linux/macOS installation in this terminal. Namely, Run the following command.
```bash
curl https://cy.tools/install.sh | bash
```

4. Enjoy CYTools! 🎉

</TabItem>
</Tabs>

You can take a look at exactly what is being done by the install script by looking at the `scripts/install.sh` file in the [GitHub repository](https://github.com/LiamMcAllisterGroup/cytools) (that's the point of open-source software!). In short, it downloads an `environment.yml` file and then creates a conda environment from this file.

## Usage

The CYTools environment will be named `cytools`. To enter this environment, type `conda activate cytools` in the WSL terminal. In this environment, `cytools` can be imported in the same way as any other Python package such as NumPy, SciPy, Matplotlib, etc. By default, JupyterLab is installed, which can simply be opened with `jupyter lab` (note: the webpage might not be automatically opened. Simply enter the URL in your browser). To exit the conda environment, simply call `conda deactivate`.

## Removal
To remove the CYTools environment, first enter the WSL terminal. In this terminal, run run `conda env remove --name cytools` or `conda env remove --name cytools-dev`, depending on which version (normal or dev) you installed. If you also want to remove the installed packages, add the flag `--all` to the end of the above commands.

## Troubleshooting

Since most of the testing for CYTools is done on Linux it is possible that our installation scripts don't always work. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
