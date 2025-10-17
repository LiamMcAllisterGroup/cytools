---
id: config
title: Configuration
---

There is a configuration submodule that allows changing a few settings for those adventurous enough to use CYTools in a non-standard way.

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
