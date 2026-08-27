"""Tests for the gateway-reload oracle.

An oracle that cannot go red verifies nothing, so the cases here are mostly
the FAILING ones: each check's named falsifying input, plus the three bugs
found by running the first version of this tool against the real machine
rather than against fixtures.
"""

from __future__ import annotations

import json
import plistlib

import pytest

from tools import verify_gateway_reload as vgr


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A minimal, correctly-wired gateway checkout."""
    root = tmp_path / "hermes-agent"
    (root / "gateway").mkdir(parents=True)
    (root / "gateway" / "source_watcher.py").write_text("x = 1\n")
    (root / "gateway" / "run.py").write_text(
        "class GatewayRunner:\n"
        "    def _start_source_watcher_task(self) -> None:\n"
        "        pass\n"
        "\n"
        "    async def start(self) -> bool:\n"
        "        self._start_source_watcher_task()\n"
        "        return True\n"
        "\n"
        "    def stop(self):\n"
        "        pass\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return root


class TestWired:
    def test_a_wired_tree_passes(self, tree):
        ok, detail = vgr.check_wired(tree)
        assert ok, detail

    def test_an_unvendored_watcher_fails(self, tree):
        (tree / "gateway" / "source_watcher.py").unlink()
        ok, detail = vgr.check_wired(tree)
        assert not ok and "not vendored" in detail

    def test_a_defined_but_uncalled_watcher_fails(self, tree):
        """The failure this feature exists to prevent, applied to itself."""
        run_py = tree / "gateway" / "run.py"
        run_py.write_text(
            run_py.read_text().replace("        self._start_source_watcher_task()\n", "")
        )
        ok, detail = vgr.check_wired(tree)
        assert not ok and "never calls it" in detail

    def test_a_call_from_a_different_method_does_not_count(self, tree):
        """Cutting at the method boundary, not scanning the whole file."""
        run_py = tree / "gateway" / "run.py"
        run_py.write_text(
            run_py.read_text()
            .replace("        self._start_source_watcher_task()\n        return True\n", "        return True\n")
            .replace("    def stop(self):\n        pass\n", "    def stop(self):\n        self._start_source_watcher_task()\n")
        )
        ok, detail = vgr.check_wired(tree)
        assert not ok and "never calls it" in detail

    def test_the_call_site_is_found_however_long_start_gets(self, tree):
        """`start` is already ~40 KB in the real file.

        The first version scanned a fixed 20 000-character window and reported
        the correctly-wired real gateway as unwired.
        """
        run_py = tree / "gateway" / "run.py"
        padding = "        _x = 1  # filler\n" * 3000
        run_py.write_text(
            run_py.read_text().replace(
                "    async def start(self) -> bool:\n",
                "    async def start(self) -> bool:\n" + padding,
            )
        )
        assert len(run_py.read_text()) > 60_000
        ok, detail = vgr.check_wired(tree)
        assert ok, detail


class TestSupervised:
    def _plist(self, path, *, label, home, keepalive=True, flag=True):
        args = ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"]
        if flag:
            args.append("--external-supervisor")
        path.write_bytes(
            plistlib.dumps(
                {
                    "Label": label,
                    "KeepAlive": keepalive,
                    "ProgramArguments": args,
                    "EnvironmentVariables": {"HERMES_HOME": str(home)},
                }
            )
        )

    @pytest.fixture
    def agents(self, tmp_path, monkeypatch):
        d = tmp_path / "Library" / "LaunchAgents"
        d.mkdir(parents=True)
        monkeypatch.setattr(vgr.Path, "home", staticmethod(lambda: tmp_path))
        return d

    def test_it_picks_the_job_for_this_hermes_home(self, tree, agents, tmp_path):
        """Found live: with a `-calcifer` profile job present, the first
        version reported on a completely different gateway's process."""
        self._plist(agents / "ai.hermes.gateway-calcifer.plist", label="calcifer", home=tmp_path / "profiles" / "calcifer")
        self._plist(agents / "ai.hermes.gateway.plist", label="mine", home=tmp_path)
        ok, detail, env = vgr.check_supervised(tree)
        assert ok and "mine" in detail

    def test_keepalive_false_fails(self, tree, agents, tmp_path):
        """A clean exit that nothing restarts is an outage, not a reload."""
        self._plist(agents / "ai.hermes.gateway.plist", label="mine", home=tmp_path, keepalive=False)
        ok, detail, _ = vgr.check_supervised(tree)
        assert not ok and "KeepAlive" in detail

    def test_a_missing_supervisor_flag_fails(self, tree, agents, tmp_path):
        self._plist(agents / "ai.hermes.gateway.plist", label="mine", home=tmp_path, flag=False)
        ok, detail, _ = vgr.check_supervised(tree)
        assert not ok and "external-supervisor" in detail

    def test_no_job_at_all_fails(self, tree, agents):
        ok, detail, _ = vgr.check_supervised(tree)
        assert not ok and "no launchd job" in detail


class TestArmed:
    def test_the_opt_out_fails_the_check(self, tree, monkeypatch):
        ok, detail = vgr.check_armed(
            tree, {"HERMES_GATEWAY_EXTERNAL_SUPERVISOR": "1", "HERMES_GATEWAY_SOURCE_RELOAD_OFF": "1"}
        )
        assert not ok and "watching is off" in detail

    def test_an_unsupervised_environment_fails_the_check(self, tree):
        ok, detail = vgr.check_armed(tree, {})
        assert not ok and "should_watch" in detail


class TestCurrent:
    def _heartbeat(self, tmp_path, *, started):
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        (state / "gateway.heartbeat").write_text(
            json.dumps({"pid": vgr.os.getpid(), "start_time": started})
        )

    def test_a_process_older_than_its_code_fails(self, tree, tmp_path, monkeypatch):
        """The actual property: a gateway running code nobody can read."""
        monkeypatch.setattr(vgr, "installed_root", lambda: tree)
        newest = max(p.stat().st_mtime for p in tree.rglob("*.py"))
        self._heartbeat(tmp_path, started=newest - (vgr.GRACE_SECONDS + 60))
        ok, detail = vgr.check_current(tree)
        assert not ok and "not what is on disk" in detail

    def test_a_change_inside_the_grace_window_passes(self, tree, tmp_path, monkeypatch):
        """Otherwise this goes red every time somebody saves a file."""
        monkeypatch.setattr(vgr, "installed_root", lambda: tree)
        newest = max(p.stat().st_mtime for p in tree.rglob("*.py"))
        self._heartbeat(tmp_path, started=newest - 10)
        ok, detail = vgr.check_current(tree)
        assert ok, detail

    def test_a_process_newer_than_its_code_passes(self, tree, tmp_path, monkeypatch):
        monkeypatch.setattr(vgr, "installed_root", lambda: tree)
        newest = max(p.stat().st_mtime for p in tree.rglob("*.py"))
        self._heartbeat(tmp_path, started=newest + 60)
        ok, detail = vgr.check_current(tree)
        assert ok and "newer than every watched file" in detail

    def test_a_dead_pid_fails(self, tree, tmp_path, monkeypatch):
        monkeypatch.setattr(vgr, "installed_root", lambda: tree)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        (state / "gateway.heartbeat").write_text(
            json.dumps({"pid": 2**30, "start_time": 1.0})
        )
        ok, detail = vgr.check_current(tree)
        assert not ok and "no live gateway" in detail

    def test_it_refuses_to_judge_a_tree_it_is_not_running(self, tree, tmp_path, monkeypatch):
        """Comparing the heartbeat against some other checkout measures nothing.

        Found live: run against a feature worktree, the first version reported
        a branch under active edit as a broken deployment.
        """
        monkeypatch.setattr(vgr, "installed_root", lambda: tmp_path / "elsewhere")
        ok, detail = vgr.check_current(tree)
        assert ok and "not the installed tree" in detail
