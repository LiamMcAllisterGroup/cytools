---
title: Configuration
---

# Configuration

There is a configuration submodule, `cytools.config`, that allows changing a
few settings for those adventurous enough to use CYTools in a non-standard way.

## Experimental features

There are a few experimental features that are locked by default since they
haven't been through enough testing. They can be enabled as follows.

```python
import cytools
cytools.config.enable_experimental_features()
```

More details can be found in the [experimental features](experimental.md)
section.

## Restricting parallelism

When running multiple scripts at the same time, it can be detrimental if each
process tries to use all available CPU threads. To limit the number of CPU
threads that CYTools uses you can set the `n_threads` variable, say to 1, as
follows.

```python
import cytools
cytools.config.n_threads = 1
```

## Mosek license

By default, Mosek looks for its license file at `~/mosek/mosek.lic`, or at the
path given by the `MOSEKLM_LICENSE_FILE` environment variable. If the license
file is stored somewhere else, you can specify its location as follows.

```python
import cytools
cytools.config.set_mosek_path("/path/to/mosek.lic")
```

See the [advanced usage](../getting-started/advanced.md#mosek) page for
instructions on how to obtain a license.

## Reference

::: cytools.config
    options:
      show_root_heading: false
      heading_level: 3
