"""
Tests for the surviving helpers in cytools.helpers.

Extracted from PR #102. Its LIL/LIL_stack (and LazyTuple) tests are
dropped: those no longer exist, having been replaced by the CSR helpers
in cb92bd9.
"""

import gzip
import os

import pytest

from cytools.helpers import basic_geometry, misc
from cytools.helpers.matrix import flatten_top


def test_flatten_top():
    lis = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    assert flatten_top(lis) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert flatten_top(lis, N=2) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert flatten_top(lis, as_list=False).tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_get_bdry_triangle():
    from cytools import Polytope

    p = Polytope([[0, 0], [3, 1], [2, 2]])
    bdry = p.get_bdry()

    # a triangle with no interior/extra points has 3 boundary edges
    assert all(isinstance(e, frozenset) for e in bdry)
    assert all(len(e) == 2 for e in bdry)
    assert bdry == {frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3})}


def test_get_bdry_matches_brute_force():
    from cytools import Polytope

    for verts in ([[0, 0], [4, 0], [0, 4]], [[-2, -1], [3, 0], [0, 3], [1, -2]]):
        p = Polytope(verts)

        # brute-force: an edge is on the boundary iff it lies in one simplex
        edges = []
        for s in p.triangulate().simplices():
            edges += [(s[0], s[1]), (s[0], s[2]), (s[1], s[2])]
        expected = {
            frozenset(e) for e in edges if sum(o == e for o in edges) == 1
        }

        assert basic_geometry.get_bdry(p) == expected


def test_zipped_pickle_roundtrip(tmp_path):
    misc.save_zipped_pickle({"a": 1}, "cache", path=str(tmp_path))
    assert misc.load_zipped_pickle("cache", path=str(tmp_path)) == {"a": 1}

    # no stray temp files left behind
    assert os.listdir(tmp_path) == ["cache.p"]


def test_load_zipped_pickle_missing(tmp_path):
    assert misc.load_zipped_pickle("nope", path=str(tmp_path)) is None


@pytest.mark.parametrize(
    "contents",
    [
        b"not gzip at all",  # -> gzip.BadGzipFile
        b"",  # -> EOFError
    ],
)
def test_load_zipped_pickle_heals_broken_cache(tmp_path, contents):
    fname = tmp_path / "cache.p"
    fname.write_bytes(contents)

    assert misc.load_zipped_pickle("cache", path=str(tmp_path)) is None
    assert not fname.exists(), "broken cache should be removed"


def test_load_zipped_pickle_heals_truncated_gzip(tmp_path):
    fname = tmp_path / "cache.p"
    misc.save_zipped_pickle(list(range(10000)), "cache", path=str(tmp_path))

    # chop off the tail -> a valid gzip header with a broken deflate stream
    full = fname.read_bytes()
    fname.write_bytes(full[: len(full) // 2])
    with pytest.raises(Exception):
        with gzip.open(fname, "rb") as f:
            f.read()

    assert misc.load_zipped_pickle("cache", path=str(tmp_path)) is None
    assert not fname.exists(), "broken cache should be removed"
