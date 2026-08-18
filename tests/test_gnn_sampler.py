import importlib.util
import os
import subprocess
import sys
import types

import pytest

from cytools import Polytope


def _has_dualgnn() -> bool:
    """
    Whether the optional dualgnn package is importable.

    This must never raise: it runs at import (i.e. collection) time, so any
    exception here turns a missing optional dependency into a collection
    error instead of a clean skip. `importlib.util.find_spec` is not
    exception-free -- it raises ValueError if something has left a bare stub
    module in `sys.modules` (a common way of blocking optional imports, since
    such a module has `__spec__ = None`), and it propagates whatever a
    third-party meta-path finder raises.
    """
    try:
        return importlib.util.find_spec("dualgnn") is not None
    except Exception:
        return False


HAS_DUALGNN = _has_dualgnn()

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

# h11=8 polytope whose 2-faces have several FRTs each, with some
# combinations failing to glue -- a single batch of NTFE draws can come
# back short of N, exercising the retry loop
p_h11_8 = Polytope(
    [
        [1, -1, -2, 1],
        [1, -1, 0, -1],
        [-1, 0, 1, 1],
        [-1, 2, 1, -1],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, -1],
    ]
)


# the optional dependency must never break collection
# ----------------------------------------------------
def test_has_dualgnn_survives_a_broken_meta_path_finder(monkeypatch):
    class ExplodingFinder:
        def find_spec(self, fullname, path=None, target=None):
            raise ImportError("boom")

    monkeypatch.setattr(sys, "meta_path", [ExplodingFinder()])
    monkeypatch.delitem(sys.modules, "dualgnn", raising=False)
    assert _has_dualgnn() is False


def test_has_dualgnn_survives_a_bare_stub_module(monkeypatch):
    # a bare module object has `__spec__ = None`, which makes
    # `importlib.util.find_spec` raise ValueError
    monkeypatch.setitem(sys.modules, "dualgnn", types.ModuleType("dualgnn"))
    assert _has_dualgnn() is False


_STUB_SITECUSTOMIZE = """
import sys
import types

sys.modules["dualgnn"] = types.ModuleType("dualgnn")
"""


def test_collection_succeeds_when_dualgnn_import_is_blocked(tmp_path):
    # end-to-end check, in a subprocess, that this file still *collects* when
    # dualgnn cannot be imported -- the failure mode reported in issue #90
    (tmp_path / "sitecustomize.py").write_text(_STUB_SITECUSTOMIZE)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            os.path.abspath(__file__),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "tests collected" in res.stdout, res.stdout + res.stderr
    assert "error" not in res.stdout.lower(), res.stdout


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
def test_seed_reproducibility():
    # same seed -> bitwise-identical heights (per device; the torch CPU
    # and CUDA generators are independent streams)
    h1, h2 = (
        p_h11_8.random_triangulations_gnn(
            N=5, N_face_triangs=5, seed=7, as_heights=True
        )
        for _ in range(2)
    )
    assert len(h1) == len(h2) > 0
    assert all((a == b).all() for a, b in zip(h1, h2))


@pytest.mark.skipif(not HAS_DUALGNN, reason="dualgnn is not installed")
def test_fills_N():
    triangs = p_h11_8.random_triangulations_gnn(
        N=10, N_face_triangs=5, seed=0
    )
    assert len(triangs) == 10
    assert len(set(triangs)) == 10


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
