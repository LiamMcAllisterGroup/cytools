"""Tests for the sparse LIL containers in cytools.helpers.matrix."""

import gzip
import os

import numpy as np
import pytest

from cytools.helpers import basic_geometry, misc
from cytools.helpers.matrix import LIL, LIL_stack, LazyTuple, flatten_top


# helpers
# -------
def make_lil(rows, width=3, dtype=int):
    """Build a LIL with the given sparse rows."""
    out = LIL(dtype=dtype, width=width)
    out.append(rows)
    return out


# LazyTuple
# ---------
def test_lazytuple_defers_and_caches():
    calls = []

    def expensive():
        calls.append(1)
        return 42

    t = LazyTuple(1, expensive)
    assert calls == []
    assert t[1] == 42
    assert t[1] == 42
    assert len(calls) == 1
    assert len(t) == 2


# LIL: cache invalidation after mutation
# --------------------------------------
def test_append_invalidates_dense_cache():
    m = make_lil([{0: 1}])
    assert m.dense().tolist() == [[1, 0, 0]]

    m.append({1: 9})
    assert sorted(m.dense().tolist()) == sorted([[1, 0, 0], [0, 9, 0]])


def test_setitem_invalidates_dense_cache():
    m = make_lil([{0: 1}])
    m.dense()

    m[0, 1] = 42
    assert m.dense().tolist() == [[1, 42, 0]]


def test_new_row_invalidates_dense_cache():
    m = make_lil([{0: 1}])
    m.dense()

    m.new_row()
    assert m.dense().shape == (2, 3)


def test_append_invalidates_sum_caches():
    m = make_lil([{0: 1}])
    # prime every sum cache
    assert m.sum() == 1
    assert m.sum(axis=0).tolist() == [1, 0, 0]
    assert m.sum(axis=1).tolist() == [1]
    assert m.sum(axis=0, dense=False) == {0: 1}

    m.append({1: 5})
    assert m.sum(axis=0).tolist() == [1, 5, 0]
    assert m.sum(axis=1).tolist() == [1, 5]
    assert m.sum() == 6
    assert m.sum(axis=0, dense=False) == {0: 1, 1: 5}


def test_setitem_invalidates_sum_caches():
    m = make_lil([{0: 1}])
    assert m.sum() == 1

    m[0, 2] = 4
    assert m.sum() == 5
    assert m.sum(axis=0).tolist() == [1, 0, 4]


def test_reindex_invalidates_caches():
    m = make_lil([{0: 1}, {2: 5}])
    assert m.dense().tolist() == [[1, 0, 0], [0, 0, 5]]
    assert m.sum(axis=0).tolist() == [1, 0, 5]

    m.reindex({0: 2, 2: 0})
    assert m.sum(axis=0).tolist() == [5, 0, 1]
    assert sorted(m.dense().tolist()) == sorted([[0, 0, 1], [5, 0, 0]])


def test_unique_rows_invalidates_caches():
    m = make_lil([{0: 1}, {0: 1}])
    assert m.sum(axis=0).tolist() == [2, 0, 0]

    m.unique_rows()
    assert len(m) == 1
    assert m.sum(axis=0).tolist() == [1, 0, 0]


# LIL: copying semantics
# ----------------------
def test_append_tocopy_deep_copies_rows():
    rows = [{0: 1}, {1: 2}]
    m = LIL(dtype=int, width=3)
    m.append(rows, tocopy=True)

    m[0, 0] = 99
    assert rows[0] == {0: 1}, "tocopy=True must not share row dicts"

    rows[1][2] = 7
    # NOTE: dense() dedupes, which may reorder the rows
    assert sorted(m.dense().tolist()) == sorted([[99, 0, 0], [0, 2, 0]])


def test_append_no_copy_shares_rows():
    rows = [{0: 1}]
    m = LIL(dtype=int, width=3)
    m.append(rows, tocopy=False)

    m[0, 0] = 99
    assert rows[0] == {0: 99}


def test_dense_tocopy():
    m = make_lil([{0: 1}])
    assert m.dense() is m.dense()
    assert m.dense(tocopy=True) is not m.dense()


# LIL: numpy protocol
# -------------------
def test_array_protocol_respects_dtype_positionally():
    m = make_lil([{0: 1}])

    # numpy 1.x calls __array__(dtype) positionally
    assert m.__array__(np.float64).dtype == np.float64
    assert np.asarray(m, dtype=np.float64).dtype == np.float64
    assert np.array(m).tolist() == [[1, 0, 0]]


def test_array_protocol_copy_is_second_arg():
    m = make_lil([{0: 1}])
    out = m.__array__(None, True)
    assert out is not m.dense()
    assert out.tolist() == [[1, 0, 0]]


# LIL: width inference
# --------------------
def test_infer_width_with_all_default_rows():
    m = LIL(dtype=int)
    m.append({})
    assert m.infer_width() == 0

    m2 = LIL(dtype=int)
    m2.append([{}, {4: 1}, {}])
    assert m2.infer_width() == 5


def test_infer_width_empty_matrix():
    assert LIL(dtype=int).infer_width() == 0


def test_dense_infers_width_when_unset():
    m = LIL(dtype=int)
    m.append([{0: 1}, {5: 2}])
    assert sorted(m.dense().tolist()) == sorted(
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2]]
    )
    assert m.shape == (2, 6)


def test_dense_with_all_default_rows():
    m = LIL(dtype=int)
    m.append([{}, {}])
    assert m.dense().shape == (1, 0)


def test_dense_nonzero_default_val():
    m = LIL(dtype=int, width=2)
    m.default_val = -1
    m.append({0: 3})
    assert m.dense().tolist() == [[3, -1]]


# LIL: misc interface
# -------------------
def test_getitem_and_bounds():
    m = make_lil([{0: 1}])
    assert m[0, 0] == 1
    assert m[0, 2] == 0
    assert m[0] == {0: 1}
    with pytest.raises(IndexError):
        m[0, 3]


def test_setitem_requires_tuple():
    m = make_lil([{0: 1}])
    with pytest.raises(ValueError):
        m[0] = 5


def test_add_concatenates_rows():
    a = make_lil([{0: 1}])
    b = make_lil([{1: 2}])
    assert sorted((a + b).dense().tolist()) == sorted([[1, 0, 0], [0, 2, 0]])


def test_col_inds_and_tolist():
    m = make_lil([{0: 1}, {2: 3}])
    assert m.col_inds() == {0, 2}
    assert sorted(m.tolist()) == sorted([[1, 0, 0], [0, 0, 3]])


def test_append_empty_is_noop():
    m = make_lil([{0: 1}])
    assert m.append([]) is m
    assert len(m) == 1


def test_iter_densely():
    m = LIL(dtype=int, width=2, iter_densely=True)
    m.append({0: 1})
    assert [r.tolist() for r in m] == [[1, 0]]

    m.iter_densely = False
    assert list(m) == [{0: 1}]


# LIL_stack
# ---------
def stack_of(*blocks):
    """A LIL_stack picking option 0 out of every block."""
    options = [[b] for b in blocks]
    n = len(blocks)
    return LIL_stack(options, [0] * n, [1] * n)


def test_stack_dense_and_shape():
    s = stack_of(make_lil([{0: 7}, {1: 5}]), make_lil([{2: 1}]))
    assert len(s) == 3
    assert s.dense().shape == (3, 3)
    assert sorted(s.tolist()) == sorted([[7, 0, 0], [0, 5, 0], [0, 0, 1]])


def test_stack_unique_rows_invalidates_dense_cache():
    s = stack_of(make_lil([{0: 7}, {0: 7}]), make_lil([{2: 1}]))
    before = s.dense()

    s.unique_rows()
    after = s.dense()
    assert after is not before, "unique_rows must invalidate the dense cache"
    assert len(after) == 2
    assert sorted(after.tolist()) == sorted([[7, 0, 0], [0, 0, 1]])


def test_stack_sum_axis0_is_per_column():
    s = stack_of(make_lil([{0: 7}, {1: 5}]), make_lil([{2: 1}]))
    assert s.sum(axis=0).tolist() == [7, 5, 1]
    assert sorted(s.sum(axis=1).tolist()) == [1, 5, 7]
    assert s.sum() == 13


def test_stack_sum_sparse_not_implemented():
    s = stack_of(make_lil([{0: 1}]))
    with pytest.raises(NotImplementedError):
        s.sum(axis=0, dense=False)


def test_stack_len_does_not_materialize_rows():
    s = stack_of(make_lil([{0: 7}, {1: 5}]), make_lil([{2: 1}]))
    assert len(s) == 3
    assert "_arr" not in s.__dict__, "__len__ should not build the row list"

    # shape/is_empty stay lazy too
    assert tuple(s.shape) == (3, 3)
    assert not s.is_empty
    assert "_arr" not in s.__dict__


def test_stack_len_tracks_materialized_rows():
    s = stack_of(make_lil([{0: 7}, {0: 7}]), make_lil([{2: 1}]))
    assert len(s) == 3

    s.unique_rows()
    assert len(s) == 2


def test_stack_is_empty():
    assert stack_of(LIL(dtype=int, width=3)).is_empty
    assert not stack_of(make_lil([{0: 1}])).is_empty


def test_stack_getitem_and_bounds():
    s = stack_of(make_lil([{0: 7}, {1: 5}]), make_lil([{2: 1}]))
    assert s[0] == {0: 7}
    assert s[2] == {2: 1}
    assert s[2, 2] == 1
    assert s[0, 1] == 0

    with pytest.raises(IndexError):
        s[3]
    with pytest.raises(IndexError):
        s[-1]
    with pytest.raises(IndexError):
        s[3, 0]
    with pytest.raises(IndexError):
        s[-1, 0]


def test_stack_choices_roundtrip():
    a, b, c = (make_lil([{0: 1}]) for _ in range(3))
    s = LIL_stack([[a, b], [c]], [1, 0], [2, 1])
    assert list(s.choices) == [1, 0]
    assert s.width == 3
    assert s.dtype == int


def test_stack_array_protocol_respects_dtype_positionally():
    s = stack_of(make_lil([{0: 1}]))
    assert s.__array__(np.float64).dtype == np.float64
    assert np.asarray(s, dtype=np.float64).dtype == np.float64


def test_stack_iter():
    s = stack_of(make_lil([{0: 1}], width=2))
    assert list(s) == [{0: 1}]

    s.iter_densely = True
    assert [r.tolist() for r in s] == [[1, 0]]


# flatten_top
# -----------
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


# basic_geometry
# --------------
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


def test_triangle_area_2x():
    assert basic_geometry.triangle_area_2x([[0, 0], [0, 1], [1, 0]]) == 1
    assert basic_geometry.triangle_area_2x([[0, 1], [1, 1], [2, 1]]) == 0


def test_ccw_and_intersect():
    assert not basic_geometry.ccw([0, 0], [1, 1], [2, 1])
    assert basic_geometry.ccw([0, 0], [2, 1], [1, 1])

    assert not basic_geometry.intersect([0, 0], [0, 1], [0, 0], [1, 0])
    assert basic_geometry.intersect([0, 0], [0, 1], [-1, 0.5], [1, 0.5])


# misc
# ----
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


def test_base10_roundtrip():
    bases = [3, 5, 2]
    for n in range(3 * 5 * 2):
        assert misc.to_base10(misc.from_base10(n, bases), bases) == n
