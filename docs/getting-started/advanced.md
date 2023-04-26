---
id: advanced
title: Advanced usage
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

We have compiled some guidelines for people that want to get the best performance out of CYTools or want to perform computationally intensive tasks such as a large scan over the Kreuzer-Skarke database. We discuss some performance tips and describe how to tinker with the Docker image to expand the functionality of CYTools.

## Performance tips

### Caching

CYTools classes cache most of the hard computations so that results don't have to be recomputed if they are needed again. While this gives a significant speedup during typical usage, one should also be aware of possible complications.

The cached data is stored as hidden attributes of the classes, which are never exposed by any function, and they should not be directly accessed unless you know what you are doing. Accidentally modifying one of these hidden attributes may result in following computations being wrong.

There is also the issue of memory management. If a large number of polytopes or triangulations need to be used, then it is good to make sure that not everything is kept in memory at the same time. For instance, let's look at the following example.

```python
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100, as_list=True)

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.toric_mori_cone()
    # Do some other computations
```

In the above example we have a list of 100 triangulations that we use to construct the corresponding Calabi-Yau hypersurfaces and do some computations. Each Triangulation object caches its corresponding CalabiYau object, which in turn caches its intersection numbers and whatever else we compute. So if our list of triangulations is too big then we might end up running out of memory. We could solve this in two ways.

1. We could use the ```clear_cache``` function that all main classes have to clear the results of any previous computation.
``` python {12}
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100, as_list=True)

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.ambient_mori_cone()
    # Do some other computations
    t.clear_cache()
```

2. (Preferable alternative) We could rewrite our code so that only one triangulation is kept in memory at a time. This is exactly why by default various functions in CYTools return generator objects instead of lists! The example that we considered can simply be fixed by removing the `as_list=True` argument.
``` python {5}
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100) # Now triangs is a generator instead of a list

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.ambient_mori_cone()
    # Do some other computations
```

### Identify equivalent Calabi-Yau hypersurfaces

It is common for multiple triangulations of a polytope to result in Calabi-Yau hypersurfaces that are equivalent. CYTools has functions designed to identify these cases when the equivalence is simple to check. In particular, it can check whether the restriction of the triangulations to codimension-2 faces of the polytope are equivalent or related by a polytope automorphism. The function to check for this equivalence is the following.
```python
cy1.is_trivially_equivalent(cy2)
# returns a boolean value
```
If the function returns `True` then the two hypersurfaces are certainly equivalent, but if it returns `False` it could still be that they are equivalent in a more complicated way.

When scanning a large set of Calabi-Yaus it is useful to check for this equivalence to make sure that you don't waste computing power on repeated computations. The set of Calabi-Yaus that are not trivially equivalent can be found simply by constructing a set of them in Python. Here is an example of how to do this.
```python {8,9}
from cytools import Polytope

p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
triangs = p.all_triangulations(as_list=True) # Remember that as_list=True is not always a good idea, but it is convenient for this example

all_cys = [t.get_cy() for t in triangs]

cys_not_triv_eq = set(all_cys)
cys_not_triv_eq = {t.get_cy() for t in triangs} # This is another way to construct the set

print(len(all_cys),len(cys_not_triv_eq))
# 102 5
```

In this example we can see that the set of Calabi-Yaus was significantly reduced. If we had iterated over all of the Calabi-Yaus we would have wasted a lot of time since we would have repeated the same computation over and over.

### Mosek

There are various computations that require finding a point in a cone. This task is particularly difficult when it is a high-dimensional problem and the cone is very narrow. The CYTools Docker image bundles some good tools to perform these computations, such as [OR-Tools](https://developers.google.com/optimization). However, for high-dimensional cases, the [Mosek](https://www.mosek.com/) optimizer is significantly faster. Therefore, if you plan to scan over a large number of Calabi-Yaus with large $h^{1,1}$, it is worth using Mosek.

Unfortunately, Mosek is proprietary software and requires a license to use it. Fortunately, it is easy to obtain and activate a free academic license and we have listed the instructions to do so here.

1. You can find the form to request a free academic license at [this link](https://www.mosek.com/products/academic-licenses/).

2. Follow the instructions to request a personal academic license. Make sure to use your university email to have the request approved quickly.

3. You will receive an email with a `mosek.lic` file. Simply copy this license file into the following path (you will need to create the `mosek` directory).

<Tabs>
<TabItem value="linux" label="Linux" default>

```
/home/YOUR_USER_NAME/mosek/mosek.lic
```

</TabItem>
<TabItem value="macos" label="macOS">

```
/Users/YOUR_USER_NAME/mosek/mosek.lic
```

</TabItem>
<TabItem value="windows" label="Windows">

```
C:\Users\YOUR_USER_NAME\mosek\mosek.lic
```

</TabItem>
</Tabs>

:::note
If you prefer to place the license file in a separate directory or if you run CYTools with a non-standard mounted location you will need to configure CYTools to look for the Mosek license at the appropriate place. To do this look at the [configuration page](../documentation/config.md).
:::

In order to check if your Mosek license is properly activated you can run the following commands.

```python
from cytools import config
config.check_mosek_license()
config.mosek_is_activated()
```
If the last function returns `True` then you are all set. Also, if you have the Mosek license in the recommended path then it is not necessary to do this check every time, as the `check_mosek_license` function is run every time that CYTools is started.

## Tinkering with the Docker image

It might be tempting to tweak or add some things to the CYTools Docker image. Here is a description of how it works so that you can do so.

### What is Docker

Docker is a tool for running software packages in containers, i.e. with their own dependencies and mostly disconnected from your host system. On Windows and macOS these containers run on a virtual machine, but on Linux they don't need a virtual machine since they can use the host kernel, making them more efficient. The main point of Docker is that one can set up a **Docker image** by using a set of instructions contained in a **Dockerfile**, and then containers can be created from these images to run the desired processes isolated from each other and from the host operating system. We use a Docker image to package CYTools since it has several dependencies that are a little tricky to get working together, and so that it can be used on almost any Linux distro, macOS, and even Windows. For more information about Docker you can visit [their website](https://www.docker.com/).

### Building a Docker image

The "recipe" for building a Docker image is specified in a file called "Dockerfile". This file contains all of the commands that will run to install all the necessary dependencies and the software itself that we want to run. You can take a look at the Dockerfile for CYTools at [this link](https://github.com/LiamMcAllisterGroup/cytools/blob/main/Dockerfile). For detailed instructions on how to write a Dockerfile you can visit [this website](https://docs.docker.com/engine/reference/builder/).

The Docker image is built with the `docker build` command. As a required parameter it takes the path to the Dockerfile. It is always useful to specify a name and tag with the `-t` option, which takes a parameter of the form `name` or `name:tag`, and if the former is used then the tag is set to "latest". Thus, the typical way to build a Docker image is to navigate to the root of the source code and run

```bash
docker build -t some-software .
```
where the dot at the end means to use the current directory.

However, to make CYTools compatible with most computers and make it play nicely with the host file system we have to input a bunch of extra parameters. We have made a table explaining each of them.

| Parameter | Description |
| ----- | -------- |
| `--no-cache` | It tells Docker to not use cached intermediate images from previous builds. This helps because sometimes there can be stale information that needs to be updated causing the build to fail. However, this has the downside of making the builds take longer. |
| `--force-rm` | It tells Docker to remove intermediate containers once it is done building the image. |
| `-t cytools:uid-[USERID]` | As discussed above, this gives the image a name and tag. We tag each image with the user ID because each image is built tailored to each user of the host machine. On Linux and macOS `[USERID]` is set to `$UID` while on Windows it is set to `0`. |
| `--build-arg USERNAME=cytools` | This specifies the name of the user in the Docker image. It is usually set to `cytools` except when building an image with root user or when using Windows, in which case it is set to `root`. |
| `--build-arg USERID=[USERID]` | This specifies what the user ID of the user in the Docker image should be. This is so that the permissions of the files play nicely with the host machine. On Linux and macOS `[USERID]` is set to `$UID` while on Windows it is set to `0`. |
| `--build-arg ARCH=[ARCH]` | This specifies the architecture of the host machine. `[ARCH]` should be either `amd64` or `arm64`. Other architectures are unsupported. |
| `--build-arg AARCH=[AARCH]` | This again specifies the architecture of the host machine, but with some other convention. `[AARCH]` should be either `x86_64` or `aarch64`. Other architectures are unsupported. |
| `--build-arg VIRTUAL_ENV=[PATH]` | This specifies the location of the python virtual environment. `[PATH]` is usually set to `/home/cytools/cytools-venv/` except for Windows or when using the root user, in which case it is set to `/opt/cytools/cytools-venv/`. |
| `--build-arg ALLOW_ROOT_ARG=[FLAG]` | This is to specify whether Jupyter lab can run as root or not. `[FLAG]` is usually set to `" "` (an empty string) except for Windows or when using the root user, in which case it is set to `"--allow-root"`. |
| `--build-arg PORT_ARG=[PORT]` | This specifies the port to be used for Jupyter lab. `[PORT]` is usually set to `$(($UID+2875))` except for Windows or when using the root user, in which case it is set to `2875`. |

Therefore, to build the Docker image manually you can navigate to the root directory of CYTools in your terminal and then run the following command.

<Tabs groupId="operating-systems">
<TabItem value="linux" label="Linux" default>

```bash
sudo docker build --no-cache --force-rm -t cytools:uid-$UID \
                  --build-arg USERNAME=cytools --build-arg USERID=$UID \
                  --build-arg ARCH=amd64 --build-arg AARCH=x86_64 \
			      --build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
                  --build-arg ALLOW_ROOT_ARG=" " \
			      --build-arg PORT_ARG=$(($UID+2875)) .
```

</TabItem>
<TabItem value="macos intel" label="macOS (Intel)">

```bash
docker build --no-cache --force-rm -t cytools:uid-$UID \
             --build-arg USERNAME=cytools --build-arg USERID=$UID \
             --build-arg ARCH=amd64 --build-arg AARCH=x86_64 \
			 --build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
             --build-arg ALLOW_ROOT_ARG=" " \
			 --build-arg PORT_ARG=$(($UID+2875)) .
```

</TabItem>
<TabItem value="macos apple" label="macOS (Apple silicon)">

```bash
docker build --no-cache --force-rm -t cytools:uid-$UID \
             --build-arg USERNAME=cytools --build-arg USERID=$UID \
             --build-arg ARCH=arm64 --build-arg AARCH=aarch64 \
			 --build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
             --build-arg ALLOW_ROOT_ARG=" " \
			 --build-arg PORT_ARG=$(($UID+2875)) .
```

</TabItem>
<TabItem value="windows" label="Windows">

```bash
docker build --no-cache --force-rm -t cytools \
             --build-arg USERNAME=root --build-arg USERID=0 \
             --build-arg ARCH=amd64 --build-arg AARCH=x86_64 \
             --build-arg VIRTUAL_ENV=/opt/cytools/cytools-venv/ \
             --build-arg ALLOW_ROOT_ARG="--allow-root" \
             --build-arg PORT_ARG=2875 .
```

</TabItem>
</Tabs>

You can always tweak the Dockerfile and add more packages if you need to, or modify the setup to fit your needs. All you have to do is rebuild the image again, and you will be ready to go. For more information about building Docker images you can visit [this website](https://docs.docker.com/engine/reference/builder/) or the many other resources available online.

### Starting a Docker container

A Docker container can be started from a specified image with the `docker run` command. The only required parameter is the name and tag of the image, and again "latest" is used if the tag is not specified. As before, there are some extra parameters that we need to use, but this time they are much simpler. Here is a table of them.

| Parameter | Description |
| ----- | -------- |
| `--rm` | This tells Docker to delete the container after it is stopped. |
| `-it` | This tells Docker to make a container with an interactive terminal session. |
| `--name cytools-uid-[USERID]` | This assigns a name to the container, so that it is easier to identify later. `[USERID]` is usually set to `$UID` or to `0` when on Windows or using the root user. |
| `-p [HOSTPORT]:[CONTAINERPORT]` | This tells Docker to attach a specified port of the host to the specified port of the container. This is done so that we can access the Jupyter server. Both `[HOSTPORT]` and `[CONTAINERPORT]` are usually set to the `[PORT]` parameter that we used to build the image. |
| `-v [HOSTPATH]:[CONTAINERPATH]` | On Linux and macOS `[HOSTDIR]` is usually set to `$HOME` while on Windows it is set to `${home}` (in PowerShell). Note that on Linux and macOS you can specify a different `[HOSTPATH]` by using the `-d` option when lanching CYTools (e.g. `cytools -d /some/path/`). On the other hand, `[CONTAINERPATH]` is set to `/home/cytools/mounted_volume` on Linux and macOS, or to `/opt/cytools/mounted_volume` on Windows or when using the root user. |

Thus, to start a CYTools container manually you can use the following command.

<Tabs groupId="operating-systems">
<TabItem value="linux" label="Linux" default>

```bash
sudo docker run --rm -it --name cytools-uid-$UID \
                -v $HOME:/home/cytools/mounted_volume \
                -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID
```

</TabItem>
<TabItem value="macos intel" label="macOS (Intel)">

```bash
docker run --rm -it --name cytools-uid-$UID \
           -v $HOME:/home/cytools/mounted_volume \
           -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID
```

</TabItem>
<TabItem value="macos apple" label="macOS (Apple silicon)">

```bash
docker run --rm -it --name cytools-uid-$UID \
           -v $HOME:/home/cytools/mounted_volume \
           -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID
```

</TabItem>
<TabItem value="windows" label="Windows">

```bash
docker run --rm -it --name cytools \
           -v ${HOME}:/home/cytools/mounted_volume \
           -p 2875:2875 cytools
```

</TabItem>
</Tabs>

After running this command, you should see some output containing a URL of the form `http://127.0.0.1:2875/?token=xxxxxxxxx` which you have to copy and paste into your web browser of choice. If everything was set up correctly, you should find a Jupyter Lab session ready to be used.

An extra parameter can be specified to launch a specific binary. For example, one can launch a bash shell in a CYTools container with the following command.
<Tabs groupId="operating-systems">
<TabItem value="linux" label="Linux" default>

```bash
sudo docker run --rm -it --name cytools-uid-$UID \
                -v $HOME:/home/cytools/mounted_volume \
                -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID bash
```

</TabItem>
<TabItem value="macos intel" label="macOS (Intel)">

```bash
docker run --rm -it --name cytools-uid-$UID \
           -v $HOME:/home/cytools/mounted_volume \
           -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID bash
```

</TabItem>
<TabItem value="macos apple" label="macOS (Apple silicon)">

```bash
docker run --rm -it --name cytools-uid-$UID \
           -v $HOME:/home/cytools/mounted_volume \
           -p $(($UID+2875)):$(($UID+2875)) cytools:uid-$UID bash
```

</TabItem>
<TabItem value="windows" label="Windows">

```bash
docker run --rm -it --name cytools \
           -v ${HOME}:/home/cytools/mounted_volume \
           -p 2875:2875 cytools bash
```

</TabItem>
</Tabs>

This can be useful if one wants to use CYTools in a terminal. This is such a common use case that there is a parameter designed to do this in Linux and macOS. All you have to do is use the `-b` option when launching CYTools.
```bash
cytools -b
```

More information about Docker containers can be found at [this link](https://docs.docker.com/engine/reference/run/).
