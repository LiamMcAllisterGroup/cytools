"""
Regression tests for crash paths in `cytools.utils`.

Each test here pins down a bug that used to make a normal call fail loudly
(NameError/TypeError/RuntimeError), silently corrupt the caller's data, or --
in the weight-system reader -- abort the whole Python interpreter.
"""

import numpy as np
import pytest

from cytools import config
from cytools.utils import (
    fetch_polytopes,
    find_new_affinely_independent_points,
    integral_nullspace,
    read_polytopes,
)

config._exp_features_enabled = True

# a weight system of a reflexive 4d polytope (P^4)
WS = "5 1 1 1 1 1"
# another one (P^4_{1,1,1,2,2})
WS2 = "7 1 1 1 2 2"


# polytope_generator, format="ws"
# ------------------------------
# Blank lines used to be handed straight to PALP, whose C code asserts on an
# empty weight system and calls abort() -- taking the interpreter with it
# (SIGABRT/exit 134). Any input ending in a newline hit this, which includes
# every `fetch_polytopes(dim=5, ...)` response.
@pytest.mark.parametrize(
    "data",
    [
        WS,  # no trailing newline
        WS + "\n",  # trailing newline -> used to abort the interpreter
        WS + "\n\n\n",  # several trailing newlines
        "\n" + WS + "\n",  # leading blank line
        "   \n" + WS + "\n",  # whitespace-only line
    ],
)
def test_ws_reader_skips_blank_lines(data):
    polys = read_polytopes(data, input_type="str", format="ws", as_list=True)

    assert len(polys) == 1
    assert polys[0].dim() == 4
    assert polys[0].is_reflexive()


def test_ws_reader_reads_several_weight_systems_from_a_string():
    data = f"{WS}\n\n{WS2}\n"
    polys = read_polytopes(data, input_type="str", format="ws", as_list=True)

    assert len(polys) == 2
    assert [p.h11(lattice="N") for p in polys] == [101, 95]


def test_ws_reader_respects_limit():
    data = f"{WS}\n{WS2}\n"
    polys = read_polytopes(data, input_type="str", format="ws", as_list=True, limit=1)

    assert len(polys) == 1


# `input_type="file"` (the default!) used to feed the *file name* to PALP
# instead of the file contents, so reading any weight-system file failed with
# "RuntimeError: PALP error".
def test_ws_reader_reads_from_a_file(tmp_path):
    path = tmp_path / "ws.txt"
    path.write_text(f"{WS}\n\n{WS2}\n")

    polys = read_polytopes(str(path), format="ws", as_list=True)

    assert len(polys) == 2
    assert [p.h11(lattice="N") for p in polys] == [101, 95]

    # and it must agree with reading the same contents as a string
    from_str = read_polytopes(
        path.read_text(), input_type="str", format="ws", as_list=True
    )
    assert [p.vertices().tolist() for p in polys] == [
        p.vertices().tolist() for p in from_str
    ]


def test_unsupported_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported format"):
        read_polytopes(WS, input_type="str", format="not-a-format", as_list=True)


# set_curve_basis
# ---------------
# The matrix branch that pads a basis with the origin column referenced an
# undefined name `t` as the dtype, so it raised NameError for every caller.
def test_set_curve_basis_matrix_without_origin_column():
    from cytools import Polytope

    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    v = p.triangulate().get_toric_variety()

    basis = v.curve_basis(as_matrix=True)
    # drop the origin column; set_curve_basis must rebuild it
    v.set_curve_basis(basis[:, 1:])

    new_basis = v.curve_basis(as_matrix=True)
    assert new_basis.dtype == np.dtype(int)
    assert new_basis.tolist() == basis.tolist()


# find_new_affinely_independent_points
# ------------------------------------
# The single-point branch referenced an undefined name `pts_trans`.
def test_new_affinely_independent_points_from_a_single_point():
    pt = [1, 2, 3]
    new_pts = find_new_affinely_independent_points([pt])

    # the result together with the input must be affinely independent and
    # must span the full ambient space
    all_pts = np.array([pt] + new_pts.tolist())
    assert len(new_pts) == 3
    assert np.linalg.matrix_rank(all_pts[1:] - all_pts[0]) == 3


# `np.asarray` does not copy an ndarray, so the in-place translation used to
# clobber the caller's points.
def test_new_affinely_independent_points_does_not_mutate_input():
    pts = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1]])
    original = pts.copy()

    new_pts = find_new_affinely_independent_points(pts)

    assert np.array_equal(pts, original)
    # sanity check: the returned point really is affinely independent
    all_pts = np.concatenate([pts, new_pts])
    assert np.linalg.matrix_rank(all_pts[1:] - all_pts[0]) == 3


# integral_nullspace
# ------------------
# A trivial nullspace produced an empty float64 array (dividing by an empty
# float gcd array), leaking floats into callers that expect integers.
def test_integral_nullspace_is_always_integral():
    trivial = integral_nullspace(np.eye(3, dtype=int))
    assert trivial.shape == (3, 0)
    assert np.issubdtype(trivial.dtype, np.integer)

    nontrivial = integral_nullspace(np.array([[2, 2, 2], [0, 0, 0]]))
    assert np.issubdtype(nontrivial.dtype, np.integer)
    # gcd reduction must still happen
    assert nontrivial.shape == (3, 2)
    assert np.array_equal(
        np.array([[2, 2, 2]]).dot(nontrivial), np.zeros((1, 2), dtype=int)
    )


# fetch_polytopes
# ---------------
# The dim-4 Euler-characteristic consistency check guarded on `h12` but used
# `h21` in the formula, so supplying h11/h12/chi crashed with a TypeError
# before any request was made. It must accept consistent input and reject
# inconsistent input, in both lattice conventions.
# (No network access: consistent input is only checked up to the point where
# the request would be made.)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(h11=2, h12=30, chi=-56, lattice="M"),
        dict(h11=30, h12=2, chi=56, lattice="N"),
        dict(h11=2, h21=30, chi=-56, lattice="M"),
    ],
)
def test_fetch_polytopes_accepts_consistent_euler_characteristic(kwargs, monkeypatch):
    sentinel = RuntimeError("request attempted")

    def no_requests(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("cytools.utils.requests.get", no_requests)

    # reaching the request means the consistency check passed
    with pytest.raises(RuntimeError) as excinfo:
        fetch_polytopes(**kwargs)
    assert excinfo.value is sentinel


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(h11=2, h12=30, chi=-57, lattice="M"),
        dict(h11=30, h12=2, chi=57, lattice="N"),
    ],
)
def test_fetch_polytopes_rejects_inconsistent_euler_characteristic(kwargs):
    with pytest.raises(ValueError, match="Inconsistent Euler characteristic"):
        fetch_polytopes(**kwargs)
