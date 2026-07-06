# Installing CYTools

Full documentation is available on the [CYTools website](https://liammcallistergroup.github.io/cytools/).

CYTools runs on Linux and Apple Silicon (M-series) macOS. Intel-based Macs are not supported.

## Choosing an installation method

```
Want a standalone CYTools application (a clickable app that opens JupyterLab)?
  yes -> Option A: run the install script
  no  -> Do you need Normaliz (e.g. for Hilbert bases)?
           no  -> Option B: pip
           yes -> Option C: conda
                    Do you want to actively modify CYTools' source code?
                      no  -> standard environment (environment.yml)
                      yes -> development environment (environment-dev.yml)
```

## Option A — standalone application (Linux and macOS)

Run the installer for your platform from a clone of this repository.

On Linux:

```bash
bash scripts/linux/install.sh
```

On macOS:

```bash
bash scripts/macos/install.sh
```

It sets up everything (including a conda environment) and installs a clickable CYTools launcher/icon. If conda is not already installed, it offers to install Miniforge.

To uninstall, run the matching uninstaller.

On Linux:

```bash
bash scripts/linux/uninstall.sh
```

On macOS:

```bash
bash scripts/macos/uninstall.sh
```

It removes the launcher and icon, and asks whether to also remove the conda environment.

## Option B — pip

Install CYTools and its dependencies from PyPI (no clone required):

```bash
pip install cytools
```

Note: the pip install does not include Normaliz; use conda (Option C) if you need it.

To uninstall:

```bash
pip uninstall cytools
```

## Option C — conda

Clone the repository:

```bash
git clone https://github.com/LiamMcAllisterGroup/cytools.git
cd cytools
```

If you do **not** plan to modify CYTools' source code, create the standard environment:

```bash
conda env create -f environment.yml
conda activate cytools
```

If you **do** want to develop CYTools (an editable install), create the development environment instead:

```bash
conda env create -f environment-dev.yml
conda activate cytools-dev
```

Then start JupyterLab with `jupyter lab`, or run `python` and `import cytools`.

To uninstall, remove the environment you created.

The standard environment:

```bash
conda env remove -n cytools
```

The development environment:

```bash
conda env remove -n cytools-dev
```
