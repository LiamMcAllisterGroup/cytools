# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  This module contains various configuration variables for
#               experimental features and custom installations.
# -----------------------------------------------------------------------------

"""Configuration options for CYTools.

This module, accessible as `cytools.config`, holds a few settings that control
how CYTools runs, as well as the switch that unlocks its experimental features.

CYTools can use the [Mosek](https://www.mosek.com/) optimizer, when it is
installed and licensed, for some optimization problems in high dimensions,
where the other available solvers can be less reliable, such as those in
[`Cone.tip_of_stretched_cone`][cytools.cone.Cone.tip_of_stretched_cone] and
[`Cone.find_interior_point`][cytools.cone.Cone.find_interior_point] for cones
in 25 or more dimensions. When Mosek is not available, an alternative
optimizer is used. By default, the license is located through Mosek's own
discovery mechanism (the `MOSEKLM_LICENSE_FILE` environment variable, or
`~/mosek/mosek.lic`); a license stored elsewhere can be specified with
[`set_mosek_path`][cytools.config.set_mosek_path].

Attributes:
    n_threads (int | None): The number of CPU threads to use in parallelized
        computations, such as finding the extremal rays of a cone with
        [`Cone.extremal_rays`][cytools.cone.Cone.extremal_rays]. If `None`,
        the number is chosen automatically: a single thread is used for small
        problems, and all available threads otherwise. When running several
        CYTools processes at the same time, it is usually best to set this to
        a small number such as 1.

Examples:
    We restrict CYTools to a single CPU thread, and then enable the
    experimental features.

    >>> import cytools
    >>> cytools.config.n_threads = 1
    >>> cytools.config.enable_experimental_features()
"""

# 'standard' imports
import warnings
import os

# The number of CPU threads to use in some computations, such as finding the
# extremal rays of a cone. When set to None, then it uses all available threads.
n_threads = None

# Mosek license
# Default: defer to Mosek's own license discovery (the MOSEKLM_LICENSE_FILE
# environment variable, or ~/mosek/mosek.lic). set_mosek_path() overrides this.
_mosek_license = None
_mosek_is_activated = None
_mosek_error = ""


def check_mosek_license(silent=False):
    """Check whether Mosek is installed and has a valid license.

    This tries to solve an empty optimization problem with Mosek, and records
    whether it succeeded, which is what
    [`mosek_is_activated`][cytools.config.mosek_is_activated] reports. If a
    custom license path was set with
    [`set_mosek_path`][cytools.config.set_mosek_path], the
    `MOSEKLM_LICENSE_FILE` environment variable is set to it first.

    The license is checked automatically the first time Mosek might be used,
    so there is usually no need to call this function directly.

    Args:
        silent (bool): Whether to suppress the printed messages. Otherwise, a
            message is printed saying either that Mosek was successfully
            activated or why it could not be used.

    Examples:
        We check the Mosek license. If Mosek is working, a success message is
        printed. Otherwise, a message explaining the problem is printed.

        >>> import cytools
        >>> cytools.config.check_mosek_license()
        Mosek was successfully activated.
    """
    global _mosek_license
    if _mosek_license is not None:
        os.environ["MOSEKLM_LICENSE_FILE"] = _mosek_license
    global _mosek_error
    global _mosek_is_activated
    try:
        import mosek

        mosek.Env().Task(0, 0).optimize()
        _mosek_is_activated = True
        if not silent:
            print("Mosek was successfully activated.")
    except ImportError:
        _mosek_error = (
            "Info: Mosek is not installed."
        )
        _mosek_is_activated = False
    except mosek.Error as e:
        _mosek_error = (
            "Info: Mosek is not activated. "
            "An alternative optimizer will be used.\n"
            f"Error encountered: {e}"
        )
        _mosek_is_activated = False
    except Exception:
        _mosek_error = (
            "Info: There was a problem with Mosek. "
            "An alternative optimizer will be used."
        )
        _mosek_is_activated = False
    if not silent:
        print(_mosek_error)


def mosek_is_activated():
    """Return whether Mosek is installed and has a valid license.

    The license is checked (silently) with
    [`check_mosek_license`][cytools.config.check_mosek_license] the first time
    this function is called, and the result is reused afterwards. Calling
    [`check_mosek_license`][cytools.config.check_mosek_license] or
    [`set_mosek_path`][cytools.config.set_mosek_path] again updates it.

    Returns:
        bool: Whether Mosek can be used.

    Examples:
        We check whether Mosek is available. The result depends on the local
        installation.

        >>> import cytools
        >>> cytools.config.mosek_is_activated()
        False
    """
    global _mosek_error
    global _mosek_is_activated
    global _printed_mosek_error
    if _mosek_is_activated is None:
        check_mosek_license(silent=True)
    return _mosek_is_activated


def set_mosek_path(path):
    """Set a custom path to the Mosek license file.

    This is needed when the license is stored in a location where Mosek does
    not find it by default. The path is exported through the
    `MOSEKLM_LICENSE_FILE` environment variable, and the license is then
    checked with [`check_mosek_license`][cytools.config.check_mosek_license],
    which prints whether Mosek could be activated.

    Args:
        path (str): The path to the Mosek license file.

    Examples:
        We set the path to a license file stored in a custom location.

        >>> import cytools
        >>> cytools.config.set_mosek_path("/path/to/mosek.lic")
    """
    global _mosek_license
    _mosek_license = path
    check_mosek_license()


# Lock experimental features by default.
_exp_features_enabled = False


def enable_experimental_features():
    """Enable the experimental features of CYTools.

    Some features are locked by default because they have not been tested
    enough, such as Calabi-Yau manifolds of dimension other than 3,
    non-favorable Calabi-Yau hypersurfaces, complete intersection Calabi-Yaus,
    and generic divisor and curve bases specified as matrices.
    This function unlocks them for the rest of the session and emits a warning,
    since these features may be broken or change significantly in future
    versions. See the Experimental Features page of the documentation for more
    details.

    Examples:
        We enable the experimental features.

        >>> import cytools
        >>> cytools.config.enable_experimental_features()
    """
    global _exp_features_enabled
    _exp_features_enabled = True
    warnings.warn(
        "\n**************************************************************\n"
        "Warning: You have enabled experimental features of CYTools.\n"
        "Some of these features may be broken or not fully tested,\n"
        "and they may undergo significant changes in future versions.\n"
        "**************************************************************\n"
    )
