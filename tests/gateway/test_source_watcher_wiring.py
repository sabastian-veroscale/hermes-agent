"""The watcher wired into GatewayRunner.

``test_source_watcher.py`` covers the watcher in isolation. This covers the
part that can only go wrong once it is attached to a live gateway:

  * arming when nothing will restart the process (the outage case);
  * the poll loop actually reaching ``request_restart`` rather than firing a
    callback into the void;
  * a failure in any of it aborting gateway startup.

These drive their own event loop rather than using ``@pytest.mark.asyncio``,
so they run under a bare pytest without the asyncio plugin installed.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

import gateway.run as gateway_run
from gateway.run import GatewayRunner
from gateway.source_watcher import EXTERNAL_SUPERVISOR_ENV


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "gateway").mkdir()
    (tmp_path / "gateway" / "run.py").write_text("x = 1\n")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_ambient_supervisor(monkeypatch):
    """Start from "unsupervised" whatever launched pytest.

    ``should_watch`` recognises systemd (``INVOCATION_ID``), s6 and launchd
    (``XPC_SERVICE_NAME``) as well as the explicit flag -- so a CI runner
    under systemd would silently satisfy the negative case and it would stop
    testing anything.
    """
    for var in (
        EXTERNAL_SUPERVISOR_ENV,
        "INVOCATION_ID",
        "HERMES_S6_SUPERVISED_CHILD",
        "XPC_SERVICE_NAME",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def runner(monkeypatch, tree):
    r = object.__new__(GatewayRunner)
    r._draining = False
    r._restart_task_started = False
    r._background_tasks = set()
    r._source_watcher_task = None
    r.restart_calls = []

    def _request_restart(*, detached=False, via_service=False):
        r.restart_calls.append({"detached": detached, "via_service": via_service})
        r._restart_task_started = True
        return True

    r.request_restart = _request_restart
    monkeypatch.setattr(gateway_run, "gateway_source_root", lambda: tree)
    return r


def _run(coro):
    """Drive a coroutine to completion on a fresh loop, cancelling leftovers."""

    async def _wrapper():
        try:
            return await coro
        finally:
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    return asyncio.run(_wrapper())


class TestArming:
    def test_start_actually_arms_it(self):
        """The failure mode this whole feature exists to prevent: code that is
        correct, tested, and never invoked.

        ``start()`` cannot be driven in a unit test -- it connects adapters --
        so this asserts the call site rather than the behaviour. Every other
        test here passes with the call site deleted; without this one, the
        watcher could ship fully working and never run.
        """
        assert "self._start_source_watcher_task()" in inspect.getsource(
            GatewayRunner.start
        )

    def test_refuses_without_a_supervisor(self, runner, monkeypatch):
        """Nothing would start the next process -- this must stay a no-op."""
        monkeypatch.delenv(EXTERNAL_SUPERVISOR_ENV, raising=False)

        async def body():
            runner._start_source_watcher_task()
            assert runner._source_watcher_task is None

        _run(body())

    def test_arms_under_a_supervisor(self, runner, monkeypatch):
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            assert runner._source_watcher_task is not None

        _run(body())

    def test_is_idempotent(self, runner, monkeypatch):
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            first = runner._source_watcher_task
            runner._start_source_watcher_task()
            assert runner._source_watcher_task is first

        _run(body())

    def test_a_broken_watcher_cannot_abort_startup(self, runner, monkeypatch):
        """Startup must survive this feature failing outright."""
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        def _explode():
            raise RuntimeError("no source root")

        monkeypatch.setattr(gateway_run, "gateway_source_root", _explode)

        async def body():
            runner._start_source_watcher_task()  # must not raise
            assert runner._source_watcher_task is None

        _run(body())

    def test_it_is_tagged_as_a_permanent_watcher(self, runner, monkeypatch):
        """Untagged, an armed idle gateway looks busy forever to scale-to-zero."""
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            task = runner._source_watcher_task
            assert getattr(task, "_hermes_supervised_watcher", False) is True
            assert task in runner._background_tasks

        _run(body())


class TestPollLoop:
    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch):
        monkeypatch.setattr(gateway_run, "SOURCE_WATCH_POLL_SECONDS", 0.01)
        monkeypatch.setattr(gateway_run, "SOURCE_WATCH_QUIET_SECONDS", 0.05)

    def test_a_settled_edit_requests_a_graceful_restart(self, runner, monkeypatch, tree):
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            (tree / "gateway" / "run.py").write_text("x = 2\n")
            await asyncio.wait_for(runner._source_watcher_task, timeout=10)

        _run(body())
        assert runner.restart_calls == [{"detached": False, "via_service": True}]

    def test_a_quiet_tree_never_restarts(self, runner, monkeypatch):
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            await asyncio.sleep(0.3)
            assert not runner._source_watcher_task.done()

        _run(body())
        assert runner.restart_calls == []

    def test_it_stands_down_once_the_gateway_is_draining(self, runner, monkeypatch, tree):
        """A restart already under way must not be disturbed by a deploy."""
        monkeypatch.setenv(EXTERNAL_SUPERVISOR_ENV, "1")

        async def body():
            runner._start_source_watcher_task()
            runner._draining = True
            (tree / "gateway" / "run.py").write_text("x = 2\n")
            await asyncio.wait_for(runner._source_watcher_task, timeout=10)

        _run(body())
        assert runner.restart_calls == []
