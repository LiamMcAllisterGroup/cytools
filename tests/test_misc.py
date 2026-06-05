import gzip
import os
import signal
import subprocess
import sys
import textwrap
import time

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
