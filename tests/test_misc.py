import gzip
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np

from cytools.helpers import misc


# save_zipped_pickle must write atomically (temp file + os.replace) so a SIGKILL
# mid-write never leaves a corrupt destination. The child parks at the swap point
# with the temp written but the rename pending, so the parent can kill it there.
def test_writer_killed_before_swap_leaves_destination_valid(tmp_path):
    misc.save_zipped_pickle({"old": True}, "g.pkl.gz", path=str(tmp_path))
    target = tmp_path / "g.pkl.gz"
    ready = tmp_path / "ready"

    child = textwrap.dedent(
        """
        import os, sys, time
        from cytools.helpers import misc
        path, ready = sys.argv[1], sys.argv[2]
        real = os.replace
        def park(src, dst):
            open(ready, "w").close()
            time.sleep(30)
            return real(src, dst)
        os.replace = park
        misc.save_zipped_pickle({"new": 123}, "g.pkl.gz", path=path)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", child, str(tmp_path), str(ready)])
    try:
        deadline = time.time() + 30
        while not ready.exists():
            assert proc.poll() is None, "writer exited before reaching the swap"
            assert time.time() < deadline, "writer never reached the swap"
            time.sleep(0.01)
        proc.kill()
    finally:
        proc.wait()

    assert proc.returncode == -signal.SIGKILL
    with gzip.open(target, "rb") as f:
        f.read()
    assert misc.load_zipped_pickle("g.pkl.gz", path=str(tmp_path)) == {"old": True}


def test_integral_nullspace_is_saturated():
    """The integral nullspace must be the saturated kernel, not a sublattice.

    Deriving it from a rational nullspace and dividing columns by their gcd
    gives column-primitive vectors, which is weaker: their span can be a proper
    finite-index sublattice of {x in Z^n : M x = 0}. Callers that complete the
    basis to a change of coordinates then get |det| > 1.
    """
    import flint

    from cytools.utils import integral_nullspace

    def elementary_divisors(rows):
        s = np.array(
            flint.fmpz_mat(np.asarray(rows, dtype=int).tolist()).snf().tolist(),
            dtype=object)
        return [int(s[i, i]) for i in range(min(s.shape)) if int(s[i, i]) != 0]

    # the concrete failure: x is in the kernel and must be in the span
    M = np.array([[9, 4, 7]])
    K = np.asarray(integral_nullspace(M))
    x = np.array([1, -4, 1])
    assert not (M @ x).any()
    coeffs = np.linalg.lstsq(K.astype(float), x.astype(float), rcond=None)[0]
    assert np.allclose(coeffs, np.rint(coeffs)), (
        f"x = {x.tolist()} is in the integral nullspace but not in the span "
        f"of the returned basis; coefficients were {coeffs.tolist()}")

    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(60):
        r = int(rng.integers(1, 5))
        c = int(rng.integers(r + 1, 9))
        A = rng.integers(-6, 7, (r, c))
        if np.linalg.matrix_rank(A) < r:
            continue
        checked += 1
        B = np.asarray(integral_nullspace(A))
        assert not (A @ B).any()
        assert all(d == 1 for d in elementary_divisors(B.T)), (
            f"non-saturated kernel for A = {A.tolist()}")
    assert checked > 20


def test_integral_nullspace_edge_cases():
    """Shapes at the boundaries: no rows, full rank, a zero row."""
    from cytools.utils import integral_nullspace

    assert np.array_equal(integral_nullspace(np.zeros((0, 4), dtype=int)),
                          np.eye(4, dtype=int))
    assert np.asarray(integral_nullspace(np.eye(4, dtype=int))).shape == (4, 0)
    assert np.asarray(integral_nullspace(np.array([[0, 0, 0]]))).shape == (3, 3)
