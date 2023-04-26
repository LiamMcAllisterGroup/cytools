---
id: windows
title: Windows
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The installation of CYTools on Windows is exclusively done using Docker. A Docker image allows us to package all of the dependencies and ensure that everything will work properly on Windows, even though most of the software was developed for Linux. We have written convenient installation and launcher scripts so that you don't have to learn how use Docker or memorize long commands to use CYTools.

## Requirements

- A recent version of Windows 10 or 11.
- Any modern x86-64 processor with hardware virtualization enabled. Other architectures might work with emulation, but there could be problems.
- A Docker Desktop installation.

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

1. Install Docker Desktop, if not already installed. Installation instructions can be found [here](https://docs.docker.com/desktop/windows/install/).

2. Download the installer script from the following link.
<p align="center">
    <a href="https://cy.tools/cytools-installer.bat"><img src={'/img/download.png'} width="200"/></a>
</p>

3. Double click on the installer script to run it. This will build the Docker image and create a shortcut in the start menu to start CYTools. You may be prompted with a sign advising you against running the script since it is from an unknown publisher, but you can get around it by clicking on "More info" and then on "Run anyway".

![Windows install](/img/windows_install.png)

4. Enjoy CYTools! 🎉

</TabItem>
<TabItem value="advanced" label="Advanced installation">

1. Install Docker Desktop, if not already installed. Installation instructions can be found [here](https://docs.docker.com/desktop/windows/install/).

2. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

3. (Optional) Modify `Dockerfile` and any other files to fit your needs.

4. Navigate to `[CYTools-root]/scripts/windows/` and double-click on the `install.bat`. This will run a script that builds the Docker image and creates a shortcut in the start menu to start CYTools. You may be prompted with a sign advising you against running the script since it is from an unknown publisher, but you can get around it by clicking on "More info" and then on "Run anyway".

![Windows install](/img/windows_install.png)

</TabItem>
</Tabs>

You can take a look at exactly what is being done by the the install script by looking at the `scripts/windows/install.bat` file in the [GitHub repository](https://github.com/LiamMcAllisterGroup/cytools) (that's the point of open-source software!). In short, it builds the Docker image, copies a launcher script to `%APPDATA%\CYTools` (this usually corresponds to `C:\Users\YOURUSERNAME\AppData\Roaming\CYTools\`) and creates shortcuts on start menu and the Desktop.

:::tip
By default, Docker Desktop only allows containers to take up to 2 GB of RAM. For most users this might be enough, but if you start experiencing random crashes then you should try increasing this limit in the Docker app.
:::

## Usage

CYTools can be started simply by clicking on the icon that should appear on your start menu or on your desktop after running the installation script. For more advanced options you can read our [advanced usage instructions](./advanced).

## Troubleshooting

| Error | Solution |
| ----- | -------- |
| The install or launcher scripts fail to start or disappear. | Some antivirus software may mark these scripts as suspicious and prevent them from running or even remove them. You can try making an exception for them for your antivirus, or you'll have to build and install CYTools without these scripts by following the [advanced usage instructions](./advanced). However, this method will require you to use very long commands to run CYTools. |
| `Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io on [::1]:53: read udp [::1]:57425->[::1]:53: read: connection refused` |  This is likely because the Docker hub servers are overloaded and are refusing connections. All you have to do is try again a few times until it works. |
| Errors during the installation or start up. | We have tried to print useful error messages when something goes wrong, so hopefully you can figure it out from there. |
| Random crashes for no apparent reason. | It might be that the Docker container is trying to take up more than 2 GB of RAM and it is being stopped. You can increase this limit in the Docker app. |

Since most of the testing for CYTools is done on Linux it is possible that our installation scripts don't always work. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
