---
id: cluster
title: Cluster installation
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

It is likely that you will eventually want to run CYTools on a cluster. The problem is that clusters generally don't allow you to use Docker, as it requires admin privileges. For this reason, we made this guide to help you set up CYTools using a Singularity container. [Singularity](https://sylabs.io/singularity/) is a containerization tool similar to Docker. It only works on Linux and is a bit more tricky to set up, but it allows for more flexible usage that doesn't require admin privileges. Most likely the cluster of your choice will have Singularity available to use.

:::tip info
At some point we want to offer a self-contained script that installs CYTools and all of its dependencies without any admin privileges. However, this is a tricky task and we haven't been able to fully do it. If you would like to help us please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
:::

## Installation instructions

<Tabs>
<TabItem value="easy" label="Easy installation" default>

The simplest way to build a CYTools Singularity image is with the pre-built Docker image that we host on Docker hub.

1. All you have to do is make sure that your cluster has Singularity installed, and run the command.
```bash
singularity build cytools.sif docker://liammcallistergroup/cytools:singularity
```


</TabItem>
<TabItem value="advanced with dh" label="Advanced installation (with Docker Hub)">

With this method you will need a separate computer where you can run Docker and has the same architecture of the cluster that you intend to use (almost certainly x86-64). You will also need to create a free account on Docker Hub. However, this has the advantage of not having to install Singularity on your separate computer, which can be a little tricky.

1. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

2. (Optional) Modify `Dockerfile` and any other files to fit your needs.

3. Build a special Docker image by navigating to the rood directory of CYTools in your terminal and running `sudo make build-with-root-user`. This will build an image called `cytools:root` that has everything installed as root (which is necessary for the Singularity image).

4. Go to the [Docker Hub website](https://hub.docker.com/), create an account and log in on your terminal by running `docker login`.

5. On Docker Hub create a new public repository called `cytools`. Then on your terminal assign a new tag to the image that you built that matches your new repository. For example `sudo docker -t cytools:root YOURUSERNAME/cytools:singularity`.

6. Finally, in the terminal of your cluster verify that singularity is installed and build the singularity image with the following command.
```bash
singularity build cytools.sif docker://liammcallistergroup/cytools:singularity
```

</TabItem>
<TabItem value="advanced without dh" label="Advanced installation (without Docker Hub)">

With this method you will need a separate computer where you can run both Docker and Singularity, and has the same architecture of the cluster that you intend to use (almost certainly x86-64). However, you will not have to create an account on Docker Hub.

1. Download the source code of the latest stable release of CYTools from the following link, and extract the *zip* or *tar.gz* file into your desired location.
<p align="center">
    <a href="https://github.com/LiamMcAllisterGroup/cytools/releases"><img src={'/img/download.png'} width="200"/></a>
</p>

2. (Optional) Modify `Dockerfile` and any other files to fit your needs.

3. Build a special Docker image by navigating to the rood directory of CYTools in your terminal and running `sudo make build-with-root-user`. This will build an image called `cytools:root` that has everything installed as root (which is necessary for the Singularity image).

4. Verify that singularity is installed and build the singularity image with the following command.
```bash
singularity build cytools.sif docker-daemon://cytools:root
```

5. Move the `cytools.sif` file to your cluster.

</TabItem>
</Tabs>

<br></br>

And that's it! You will end up with a `cytools.sif` file that contains all of CYTools. 

## Usage

The full documentation on how to use Singularity containers can be found at [this link](https://sylabs.io/docs/). For example, if you want to start a Python shell to do some computations with CYTools you can run
```bash
singularity exec cytools.sif python3
```

Since the CYTools is primarily tested with Docker it is possible that some functionality doesn't work properly with Singularity. If this is the case, please let us know by emailing us at [support@cy.tools](mailto:support@cy.tools).
