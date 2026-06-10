import importlib.util
import sys

import pytest

from cytools import Polytope

HAS_DUALGNN = importlib.util.find_spec("dualgnn") is not None

# dual of the quintic simplex: its ten 21-point 2-faces exercise the GNN,
# but random per-face FRTs rarely jointly extend (~0.4%), so it's only used
# for face-level tests
quintic_dual = Polytope(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, -1],
    ]
).dual()

# small (h11=2) polytope with high NTFE-extension rates, for end-to-end tests
p11169 = Polytope(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -1],
        [-1, -1, -6, -9],
    ]
)


def test_dualgnn_is_an_allowed_method():
    with pytest.raises(ValueError, match="dualgnn"):
        quintic_dual.face_triangs(triang_method="not-a-method")


def test_rejects_non_reflexive_or_wrong_dim():
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(NotImplementedError):
        p.random_triangulations_gnn(N=1)


def _block_dualgnn(monkeypatch):
    # makes `import dualgnn` raise ImportError, even if it is installed
    monkeypatch.setitem(sys.modules, "dualgnn", None)
    monkeypatch.setitem(sys.modules, "dualgnn.model", None)


def test_missing_dualgnn_noninteractive(monkeypatch):
    from cytools.ntfe import face_triangulations as ft

    _block_dualgnn(monkeypatch)
    monkeypatch.setattr(ft, "_is_interactive", lambda: False)
    # max_npts=0 forces sampling (rather than enumeration) on every 2-face
    with pytest.raises(ImportError, match="dualgnn"):
        quintic_dual.face_triangs(triang_method="dualgnn", max_npts=0)


def test_missing_dualgnn_install_declined(monkeypatch):
    from cytools.ntfe import face_triangulations as ft

    _block_dualgnn(monkeypatch)
    monkeypatch.setattr(ft, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    calls = []
    monkeypatch.setattr(ft.subprocess, "check_call", lambda cmd: calls.append(cmd))
    with pytest.raises(ImportError, match="dualgnn"):
        quintic_dual.face_triangs(triang_method="dualgnn", max_npts=0)
    assert calls == []


def test_missing_dualgnn_install_accepted(monkeypatch):
    from cytools.ntfe import face_triangulations as ft

    _block_dualgnn(monkeypatch)
    monkeypatch.setattr(ft, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    calls = []
    monkeypatch.setattr(ft.subprocess, "check_call", lambda cmd: calls.append(cmd))
    # the (mocked) install can't actually make dualgnn importable here, so
    # the re-import still fails -- but pip must have been invoked
    with pytest.raises(ImportError):
        quintic_dual.face_triangs(triang_method="dualgnn", max_npts=0)
    assert calls == [[sys.executable, "-m", "pip", "install", "dualgnn"]]


@pytest.mark.skipif(not HAS_DUALGNN, reason="dualgnn is not installed")
def test_face_triangs():
    face_triangs = quintic_dual.face_triangs(
        triang_method="dualgnn", max_npts=0, N_face_triangs=5, seed=0
    )
    assert len(face_triangs) == len(quintic_dual.faces(2))
    for f_triangs in face_triangs:
        assert 0 < len(f_triangs) <= 5
        for t in f_triangs:
            assert t.is_fine()
            assert t.is_regular()


@pytest.mark.skipif(not HAS_DUALGNN, reason="dualgnn is not installed")
def test_sample_frsts():
    triangs = p11169.random_triangulations_gnn(
        N=4, max_npts=0, N_face_triangs=5, seed=0
    )
    assert 0 < len(triangs) <= 4
    for t in triangs:
        assert t.is_fine()
        assert t.is_star()
        assert t.is_regular()


@pytest.mark.skipif(not HAS_DUALGNN, reason="dualgnn is not installed")
def test_sample_heights():
    heights = p11169.random_triangulations_gnn(
        N=4, max_npts=0, N_face_triangs=5, as_heights=True, seed=0
    )
    assert 0 < len(heights) <= 4
    for h in heights:
        assert len(h) == len(p11169.labels)
        t = p11169.triangulate(heights=h, make_star=True)
        assert t.is_fine() and t.is_star()
