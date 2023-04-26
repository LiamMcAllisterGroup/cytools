---
id: macos
title: macOS
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The installation of CYTools on macOS is exclusively done using Docker. A Docker image allows us to package all of the dependencies and ensure that everything will work properly without having to worry about compatibility issues with macOS. We have written convenient installation and launcher scripts so that you don't have to learn how use Docker or memorize long commands to use CYTools.

## Requirements

- Any recent version of macOS (>=10.13).
- Any modern Apple computer (either x86-64 or Apple silicon). Hackintosh systems might work, but there could be problems.
- A Docker Desktop installation.
- Admin access to run scripts with `sudo`.

:::note
If you don't have admin access then you can look at [these instructions](./advanced) to try to manually build the CYTools image. However, this method will require you to use very long commands to run CYTools.
:::

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

1. Install Docker Desktop, if not already installed. Installation instructions can be found [here](https://docs.docker.com/docker-for-mac/install/). Be sure to download the right version depending on whether your Mac has an Intel chip or an Apple chip.

2. Run the following command on your terminal.
```bash
curl https://cy.tools/install.sh | bash
```

3. Enjoy CYTools! 🎉

</TabItem>
<TabItem value="advanced" label="Advanced installation">

1. Install Docker Desktop, if not already installed. Installation instructions can be found [here](https://docs.docker.com/docker-for-mac/install/). Be sure to download the right version depending on whether your Mac has an Intel chip or an Apple chip.

2. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

3. (Optional) Modify `Dockerfile` and any other files to fit your needs.

4. Navigate to the root directory of CYTools in your terminal, and run `make install` . This will run a script that builds the Docker image and creates a shortcut in Launchpad to start CYTools.

</TabItem>
</Tabs>

You can take a look at exactly what is being done by the install script by looking at the `scripts/macos/install.sh` file in the [GitHub repository](https://github.com/LiamMcAllisterGroup/cytools) (that's the point of open-source software!). In short, it builds the Docker image, copies a launcher script to `/usr/local/bin/cytools` and copies a few files to make the shortcut for Launchpad.

:::tip
By default, Docker Desktop only allows containers to take up to 2 GB of RAM. For most users this might be enough, but if you start experiencing random crashes then you should try increasing this limit in the Docker app.
:::

## Usage

CYTools can be started simply by clicking on the icon that should appear in Launchpad after running the installation script. Alternatively, it can be launched by running the command `cytools` in your terminal. It can be launched into bash mode with the `-b` flag, and you can mount a custom path with the `-d [PATH]` option. For more advanced options you can read our [advanced usage instructions](./advanced).

## Troubleshooting

| Error | Solution |
| ----- | -------- |
| `bash: docker: command not found` | You must have Docker installed before you install CYTools. |
| `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?` | You have to start the Docker daemon. All it may take is to open Docker Desktop, but if you installed Docker in a different way you'll have to look it up yourself. |
| `Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io on [::1]:53: read udp [::1]:57425->[::1]:53: read: connection refused` |  This is likely because the Docker hub servers are overloaded and are refusing connections. All you have to do is try again a few times until it works. |
| `... permission denied` | Some of the commands in the install script need to be run with `sudo`. If you don't have admin access then you can look at [these instructions](./advanced) to try to manually build the CYTools image. However, this method will require you to use very long commands to run CYTools.  |
| `make: *** No rule to make target 'install'.  Stop.` or `make: *** No targets specified and no makefile found.  Stop.` | You first have to navigate to the root directory of CYTools before running the install command. |
| Random crashes for no apparent reason. | It might be that the Docker container is trying to take up more than 2 GB of RAM and it is being stopped. You can increase this limit in the Docker app. |

Since most of the testing for CYTools is done on Linux it is possible that our installation scripts don't always work. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
