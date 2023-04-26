---
id: config
title: Configuration
---

There is a configuration submodule that allows changing a few settings for those adventurous enough to use CYTools in a non-standard way, such as using it with a Singularity image, having a custom installation without a Docker image, or if you want to try some experimental features.

## Experimental Features

There are a few experimental features that are locked by default since they haven't been through enough testing. They can be enabled as follows.
```python
import cytools
cytools.config.enable_experimental_features()
```
More details can be found in the [experimental features](./experimental) section.

## Restricting parallelism

When running multiple scripts at the same time, it can be detrimental if each process tries to use all available CPU threads. To limit the number of CPU threads that CYTools uses you can set the `n_threads` variable, say to 1, as follows.
```python
import cytools
cytools.config.n_threads = 1
```

## Paths to External Software

The paths to external software are stored in the following variables of the `config` submodule.
- `cgal_path`: The directory that contains the CGAL binaries.
- `topcom_path`: The directory that contains the TOPCOM binaries.
- `palp_path`: The directory that contains the PALP binaries.

These can be changed in the following way.

```python
import cytools
cytools.config.cgal_path = "/your/custom/path/"
```
As with the Mosek license, you have to be aware that the path on your host machine will in general be different from the path in the container.

## Using Mosek with Singularity or with a non-standard path

There can be cases when we need to activate Mosek with the license file placed in a non-standard location, such as when using a singularity image. To do this we need to point CYTools to look for the image in the right path. We do so as follows. Let's say that you have your Mosek license in `/home/YOURUSERNAME/Documents/mosek/mosek.lic`. Recall that by default the CYTools script mounts `/home/YOURUSERNAME/` to `/home/cytools/mounted_volume/` in the Docker container. Thus, the path to the license in the container will be `/home/cytools/mounted_volume/Documents/mosek/mosek.lic`. However, note that when using Singularity you will need to use the normal path of your Mosek license. The way we set the license path within CYTools is as follows.

```python
import cytools
cytools.config.set_mosek_path("/home/cytools/mounted_volume/Documents/mosek/mosek.lic")
```
Once we run the above commands it will tell us if there was a problem with the Mosek license. If no message is shown then it means that it was successfully activated.

:::note
If you are also using a custom mount path for either the host or the container then you need to make sure that the location of the license is accessible from the container and that you use the right path. For more discussion about how volume mouting works you can see the [advanced usage page](../getting-started/advanced) or the [Docker documentation](https://docs.docker.com/engine/reference/run/).
:::
