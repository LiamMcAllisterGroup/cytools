---
id: linux
title: Linux
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The installation of CYTools on Linux is primarily done using Docker. A Docker image allows us to package all of the dependencies and ensure that everything will work properly regardless of the Linux distribution. We have written convenient installation and launcher scripts so that you don't have to learn how use Docker or memorize long commands to use CYTools.

## Requirements

- Almost any Linux distribution.
- A modern x86-64 processor with hardware virtualization enabled. Other architectures might work with emulation, but there could be problems.
- A Docker Engine installation.
- Admin access to run scripts with `sudo`.

:::note
If you don't have admin access or are unable to use Docker, as is the case in most clusters, then you can follow [these instructions](./cluster) to install CYTools using Singularity. If you are eager enough you can also download, compile, and install all CYTools dependencies without `sudo` access.
:::

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

1. Install Docker Engine, if not already installed. Installation instructions can be found [here](https://docs.docker.com/engine/install/). You could install Docker Desktop instead, but this is not recommended as it uses a virtual machine and requires more system resources.

2. Run the following command on your terminal.
```bash
curl https://cy.tools/install.sh | bash
```

3. Enjoy CYTools! 🎉

</TabItem>
<TabItem value="advanced" label="Advanced installation">

1. Install Docker Engine, if not already installed. Installation instructions can be found [here](https://docs.docker.com/engine/install/). You could install Docker Desktop instead, but this is not recommended as it uses a virtual machine and requires more system resources.

2. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

3. (Optional) Modify `Dockerfile` and any other files to fit your needs.

4. Navigate to the root directory of CYTools in your terminal, and run `make install` . This will run a script that builds the Docker image and creates a shortcut in the start menu to easily start CYTools.

</TabItem>
</Tabs>

You can take a look at exactly what is being done by the install script by looking at the `scripts/linux/install.sh` file in the [GitHub repository](https://github.com/LiamMcAllisterGroup/cytools) (that's the point of open-source software!). In short, it builds the Docker image, copies a launcher script to `/usr/local/bin/cytools` and copies a couple of files to make the shortcut for the start menu.

## Usage

CYTools can be started simply by clicking on the icon that should appear on your start menu after running the installation script. Alternatively, it can be launched by running the command `cytools` in your terminal. It can be launched into bash mode with the `-b` flag, and you can mount a custom path with the `-d [PATH]` option. For more advanced options you can read our [advanced usage instructions](./advanced).

## Troubleshooting

| Error | Solution |
| ----- | -------- |
| `bash: docker: command not found` | You must have Docker installed before you install CYTools. |
| `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?` | You have to start the Docker daemon. This may be a bit different between Linux distros. You can find more information at [this link](https://docs.docker.com/engine/install/linux-postinstall/). |
| `Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io on [::1]:53: read udp [::1]:57425->[::1]:53: read: connection refused` |  This is likely because the Docker hub servers are overloaded and are refusing connections. All you have to do is try again a few times until it works. |
| `... permission denied` | Some of the commands in the install script need to be run with `sudo`. If you don't have admin access or are unable to use Docker then you can follow [these instructions](./cluster) to install CYTools using Singularity. If you are eager enough you can also download, compile, and install all CYTools dependencies without using `sudo`.  |
| `make: *** No rule to make target 'install'.  Stop.` or `make: *** No targets specified and no makefile found.  Stop.` | You first have to navigate to the root directory of CYTools before running the install command. |

Since there are a many different Linux distributions it is possible that our installation scripts don't always work. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
