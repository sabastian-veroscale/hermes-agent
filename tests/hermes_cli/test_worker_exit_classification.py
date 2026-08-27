"""Worker exit statuses must survive to ``_classify_worker_exit``.

WHY THIS FILE EXISTS
--------------------
The dispatcher spawned workers with ``subprocess.Popen(...)`` and returned only
``proc.pid``, abandoning the handle. Python's ``subprocess`` module then keeps
abandoned handles in its private ``_active`` list and reaps them from
``_cleanup()``, which runs at the top of EVERY ``Popen`` construction anywhere
in the process — including the ``subprocess.run(["ps", ...])`` zombie probe
inside ``_pid_alive``. So the liveness check immediately before
``_classify_worker_exit`` was consuming the exact status that call needed.

Consequence, measured on the ops board over the 7 days to 2026-08-27:

    crashed runs                                        268
      ...carrying "pid <n> not alive"  (unknown)        177   (66%)
      ...correctly classified clean_exit                 63

A worker that merely forgot to call ``kanban_complete`` was recorded as having
crashed, handed a wrong error text on retry, and counted toward
``failure_limit`` — so the dedicated protocol-violation path was mostly dead
code and 1412 cards ended up ``blocked``.

FALSIFIER
---------
``test_abandoned_handle_loses_the_status_this_is_the_bug`` reproduces the old
behaviour directly: abandon a Popen, force ``subprocess._cleanup()`` the way a
``ps`` probe would, and assert the status is gone. If that test ever passes
trivially, the reproduction has stopped reproducing and the rest of this file
proves nothing.

Revert ``_register_worker_proc`` out of ``_default_spawn`` and
``test_retained_handle_survives_a_competing_cleanup`` goes RED.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

from hermes_cli import kanban_db  # noqa: E402


def _spawn_exiting(code=0):
    """A child that exits immediately with ``code``."""
    return subprocess.Popen(
        [sys.executable, "-c", f"raise SystemExit({code})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _force_subprocess_cleanup():
    """Do what any unrelated subprocess call in the process would do.

    ``subprocess.run(["ps", ...])`` — i.e. the zombie probe in ``_pid_alive``
    — constructs a Popen, and ``Popen.__init__`` calls ``_cleanup()`` first.
    This is that, without depending on ``ps`` being present.
    """
    subprocess.run(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )


class TestExitStatusSurvives(unittest.TestCase):

    def setUp(self):
        kanban_db._recent_worker_exits.clear()
        kanban_db._live_worker_procs.clear()

    def tearDown(self):
        kanban_db._recent_worker_exits.clear()
        kanban_db._live_worker_procs.clear()

    # -- the bug, reproduced ------------------------------------------------

    @unittest.skipIf(os.name == "nt", "POSIX reaping semantics")
    def test_abandoned_handle_loses_the_status_this_is_the_bug(self):
        proc = _spawn_exiting(0)
        pid = proc.pid
        proc.wait()          # child is done; status is inside THIS handle
        del proc             # abandon it -> lands in subprocess._active
        _force_subprocess_cleanup()   # a `ps` probe elsewhere reaps it

        kanban_db.reap_worker_zombies()
        kind, _code = kanban_db._classify_worker_exit(pid)
        self.assertEqual(
            kind, "unknown",
            "the old failure mode no longer reproduces - if abandoning a "
            "handle now preserves the status, this whole file is moot and "
            "should be deleted rather than left passing for the wrong reason",
        )

    # -- the fix ------------------------------------------------------------

    @unittest.skipIf(os.name == "nt", "POSIX reaping semantics")
    def test_retained_handle_survives_a_competing_cleanup(self):
        proc = _spawn_exiting(0)
        pid = proc.pid
        kanban_db._register_worker_proc(proc)   # what _default_spawn now does
        proc.wait()
        _force_subprocess_cleanup()             # same competing reap as above

        kanban_db.reap_worker_zombies()
        kind, code = kanban_db._classify_worker_exit(pid)
        self.assertEqual(
            kind, "clean_exit",
            "a retained handle still lost its status - the dispatcher will "
            "keep reporting protocol violations as crashes",
        )
        self.assertEqual(code, 0)

    @unittest.skipIf(os.name == "nt", "POSIX reaping semantics")
    def test_nonzero_exit_is_not_flattened_to_clean(self):
        proc = _spawn_exiting(3)
        pid = proc.pid
        kanban_db._register_worker_proc(proc)
        proc.wait()

        kanban_db.reap_worker_zombies()
        kind, code = kanban_db._classify_worker_exit(pid)
        self.assertEqual(kind, "nonzero_exit")
        self.assertEqual(code, 3)

    @unittest.skipIf(os.name == "nt", "POSIX reaping semantics")
    def test_signalled_worker_is_still_a_real_crash(self):
        """OOM-kill must NOT be reclassified as a tidy protocol violation."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid = proc.pid
        kanban_db._register_worker_proc(proc)
        proc.kill()
        proc.wait()

        kanban_db.reap_worker_zombies()
        kind, code = kanban_db._classify_worker_exit(pid)
        self.assertEqual(kind, "signaled")
        self.assertEqual(code, 9)

    # -- encoding ----------------------------------------------------------

    def test_returncode_reencoding_round_trips(self):
        """poll() gives decoded status; the classifier speaks raw. Both ways."""
        for rc, wif, want in (
            (0, os.WIFEXITED, 0),
            (1, os.WIFEXITED, 1),
            (137, os.WIFEXITED, 137),
            (-9, os.WIFSIGNALED, 9),
            (-15, os.WIFSIGNALED, 15),
        ):
            with self.subTest(rc=rc):
                raw = kanban_db._raw_status_from_returncode(rc)
                self.assertTrue(wif(raw), f"rc={rc} re-encoded to the wrong kind")
                got = os.WEXITSTATUS(raw) if rc >= 0 else os.WTERMSIG(raw)
                self.assertEqual(got, want)

    # -- the registry must not grow ----------------------------------------

    def test_registry_is_bounded_by_live_workers_not_history(self):
        """A leak here would be a slow memory bug in a long-lived gateway."""
        for _ in range(5):
            proc = _spawn_exiting(0)
            kanban_db._register_worker_proc(proc)
            proc.wait()
        kanban_db.reap_worker_zombies()
        self.assertEqual(
            kanban_db._live_worker_procs, {},
            "exited workers were not released from the retain registry",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
