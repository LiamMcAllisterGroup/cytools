---
id: linuxmacos
title: Linux/macOS
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The installation of CYTools on Linux or macOS is primarily done using conda. A conda environment allows us to package all of the dependencies and ensure that everything will work properly. In theory, a pip install would also suffice, but dependencies can be a bit cumbersome.

## Requirements

- Almost any Linux distribution or any recent version of macOS (>=10.13).
- Linux: A modern x86-64 processor with hardware virtualization enabled. Other architectures might work with emulation, but there could be problems.
- macOS: Any modern Apple computer (with an Apple silicon processor).
- (optionally but highly recommended) A conda installation. [Miniforge](https://conda-forge.org/) is recommended.

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

1. Install conda, if not already installed. Installation instructions can be found [here](https://conda-forge.org/download/).

2. Run the following command on your terminal.
```bash
curl https://cy.tools/install.sh | bash
```

3. Enjoy CYTools! 🎉

</TabItem>
<TabItem value="advanced" label="Advanced installation">

1. Install conda, if not already installed. Installation instructions can be found [here](https://conda-forge.org/download/).

2. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

3. (Optional) Modify `environment-dev.yml` and any other files to fit your needs.

4. Navigate to the root directory of CYTools in your terminal, and run `conda env create -f environment-dev.yml`. This will create a conda environment with cytools, as well as its dependencies.

</TabItem>
</Tabs>

You can take a look at exactly what is being done by the install script by looking at the `scripts/install.sh` file in the [GitHub repository](https://github.com/LiamMcAllisterGroup/cytools) (that's the point of open-source software!). In short, it downloads an `environment.yml` file and then creates a conda environment from this file.

## Usage

The CYTools environment will be named either `cytools` if the standard install was performed or `cytools-dev` if the advanced installation was followed. To enter this environment, type `conda activate cytools` or `conda activate cytools-dev`, respectively. In this environment, `cytools` can be imported in the same way as any other Python package such as NumPy, SciPy, Matplotlib, etc. By default, JupyterLab is installed, which can simply be opened with `jupyter lab`. To exit the conda environment, simply call `conda deactivate`.

## Removal
To remove the CYTools environment, run `conda env remove --name cytools` or `conda env remove --name cytools-dev`, depending on which version (normal or dev) you installed. If you also want to remove the installed packages, add the flag `--all` to the end of the above commands.

## Troubleshooting

Since there are a many different platforms it is possible that our installation scripts don't always work. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
