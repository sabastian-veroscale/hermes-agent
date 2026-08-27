"""Tests for the gateway source watcher.

The safety property under test is not "does it notice a change" -- that part is
easy and would pass with a stub. It is the two ways this feature turns into an
outage:

  * arming without a supervisor, so a restart request is just the gateway
    exiting mid-session because someone saved a file;
  * firing mid-burst, so a `git checkout` that is 40% applied gets restarted
    into.

Both have an explicit test below, and both go red if the guard is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.source_watcher import (
    DISABLE_ENV,
    EXTERNAL_SUPERVISOR_ENV,
    SourceWatcher,
    fingerprint,
    iter_source_files,
    should_watch,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "gateway").mkdir()
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "gateway" / "run.py").write_text("x = 1\n")
    (tmp_path / "hermes_cli" / "kanban_db.py").write_text("y = 1\n")
    return tmp_path


class TestShouldWatch:
    def test_refuses_without_a_supervisor(self):
        """The property that keeps an edit from becoming an outage."""
        assert should_watch({}) is False

    def test_arms_under_a_supervisor(self):
        assert should_watch({EXTERNAL_SUPERVISOR_ENV: "1"}) is True

    def test_operator_can_opt_out(self):
        assert should_watch({EXTERNAL_SUPERVISOR_ENV: "1", DISABLE_ENV: "1"}) is False

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
    def test_only_truthy_supervisor_values_arm_it(self, value):
        assert should_watch({EXTERNAL_SUPERVISOR_ENV: value}) is False

    @pytest.mark.parametrize(
        "env",
        [
            {"INVOCATION_ID": "abc"},  # systemd
            {"HERMES_S6_SUPERVISED_CHILD": "1"},  # s6
            {"XPC_SERVICE_NAME": "ai.hermes.gateway"},  # launchd
        ],
    )
    def test_it_recognises_every_supervisor_the_restart_path_does(self, env):
        """Same predicate as the existing restart routing, not a second one.

        Two definitions of "am I supervised" drifting apart is how the gate
        layer ended up with four disagreeing copies of one rule.
        """
        assert should_watch(env) is True

    def test_the_opt_out_beats_every_supervisor(self):
        assert should_watch({"INVOCATION_ID": "abc", DISABLE_ENV: "1"}) is False


class TestFingerprint:
    def test_stable_across_calls(self, tree: Path):
        assert fingerprint(tree) == fingerprint(tree)

    def test_content_change_moves_it(self, tree: Path):
        before = fingerprint(tree)
        (tree / "hermes_cli" / "kanban_db.py").write_text("y = 2\n")
        assert fingerprint(tree) != before

    def test_touch_without_content_change_does_not(self, tree: Path):
        """A checkout away and back rewrites mtimes but restores the bytes.

        Restarting a live gateway for that would be a restart for nothing,
        which is why this hashes content rather than stamping mtime.
        """
        before = fingerprint(tree)
        target = tree / "gateway" / "run.py"
        target.write_text(target.read_text())
        assert fingerprint(tree) == before

    def test_deletion_is_a_change(self, tree: Path):
        before = fingerprint(tree)
        (tree / "gateway" / "run.py").unlink()
        assert fingerprint(tree) != before

    def test_tests_and_caches_are_not_watched(self, tree: Path):
        """Editing a test cannot change what the running gateway does."""
        before = fingerprint(tree)
        (tree / "gateway" / "tests").mkdir()
        (tree / "gateway" / "tests" / "test_thing.py").write_text("assert True\n")
        (tree / "gateway" / "__pycache__").mkdir()
        (tree / "gateway" / "__pycache__" / "run.py").write_text("compiled\n")
        assert fingerprint(tree) == before

    def test_only_watched_packages_count(self, tree: Path):
        before = fingerprint(tree)
        (tree / "docs").mkdir()
        (tree / "docs" / "notes.py").write_text("z = 1\n")
        assert fingerprint(tree) == before

    def test_iter_is_deterministic(self, tree: Path):
        assert list(iter_source_files(tree)) == list(iter_source_files(tree))


class TestSourceWatcher:
    def _watcher(self, tree: Path, fired: list):
        return SourceWatcher(root=tree, on_change=fired.append, quiet_seconds=5.0)

    def test_quiet_tree_never_fires(self, tree: Path):
        fired: list = []
        watcher = self._watcher(tree, fired)
        for t in range(0, 100, 5):
            assert watcher.tick(float(t)) is False
        assert fired == []

    def test_waits_for_the_tree_to_settle(self, tree: Path):
        """A burst must produce ONE restart, after it stops -- not one per file.

        `git checkout` rewrites hundreds of files over several seconds, and a
        partially-applied tree is the worst possible state to restart into.
        """
        fired: list = []
        watcher = self._watcher(tree, fired)

        (tree / "gateway" / "run.py").write_text("x = 2\n")
        assert watcher.tick(0.0) is False, "first sight of a change must not fire"

        (tree / "hermes_cli" / "kanban_db.py").write_text("y = 2\n")
        assert watcher.tick(3.0) is False, "still changing — the burst is not over"

        assert watcher.tick(6.0) is False, "quiet period restarts on each change"
        assert watcher.tick(9.0) is True, "stable for the quiet period — fire"
        assert len(fired) == 1

    def test_fires_at_most_once(self, tree: Path):
        fired: list = []
        watcher = self._watcher(tree, fired)
        (tree / "gateway" / "run.py").write_text("x = 2\n")
        watcher.tick(0.0)
        assert watcher.tick(10.0) is True

        (tree / "gateway" / "run.py").write_text("x = 3\n")
        assert watcher.tick(20.0) is False, "already draining; a second request is noise"
        assert len(fired) == 1

    def test_a_reverted_edit_does_not_fire(self, tree: Path):
        """Save, undo, save again is an editor's normal behaviour."""
        fired: list = []
        watcher = self._watcher(tree, fired)
        original = (tree / "gateway" / "run.py").read_text()

        (tree / "gateway" / "run.py").write_text("x = 2\n")
        watcher.tick(0.0)
        (tree / "gateway" / "run.py").write_text(original)
        assert watcher.tick(3.0) is False
        assert watcher.tick(30.0) is False, "back to the running code — nothing to do"
        assert fired == []

    def test_a_failing_callback_does_not_kill_the_watcher(self, tree: Path):
        def boom(_digest: str) -> None:
            raise RuntimeError("restart refused")

        watcher = SourceWatcher(root=tree, on_change=boom, quiet_seconds=1.0)
        (tree / "gateway" / "run.py").write_text("x = 2\n")
        watcher.tick(0.0)
        assert watcher.tick(5.0) is True, "the tick reports it fired, not that it worked"

    def test_the_digest_is_handed_to_the_callback(self, tree: Path):
        fired: list = []
        watcher = self._watcher(tree, fired)
        (tree / "gateway" / "run.py").write_text("x = 2\n")
        watcher.tick(0.0)
        watcher.tick(10.0)
        assert fired == [fingerprint(tree)]
