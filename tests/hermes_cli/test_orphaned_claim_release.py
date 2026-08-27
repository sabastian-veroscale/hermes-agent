"""An expired claim on a non-running task must not deadlock the card.

WHY THIS FILE EXISTS
--------------------
Two queries have to agree about a claim lock, and they disagree by exactly one
row:

    release_stale_claims:  WHERE status = 'running'
    dispatcher ready scan: WHERE claim_lock IS NULL

A task sitting in ``ready`` while still holding a claim satisfies neither. The
reclaimer skips it (not running), the dispatcher skips it (claimed), and
nothing else looks at it. The card is stuck permanently, and — this is the part
that made it expensive — with no diagnostic anywhere. Every status query
reports a board that has simply run out of work.

Found live 2026-08-27: t_b6819a66 and t_73042427 held expired locks from
``Fortuna.local:29043`` dated 2026-08-26 00:36. They were the whole spawnable
backlog (the other two ready cards name non-Hermes profiles), so the board had
dispatched nothing for 38 hours.

FALSIFIER
---------
Remove the ``release_orphaned_claims`` call from ``dispatch_once`` and
``test_dispatcher_can_see_a_previously_deadlocked_card`` goes RED — it asserts
against the dispatcher's own output, not against the helper, so a helper that
works but is never called does not pass it.
"""

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _stick_a_claim_on(conn, task_id, *, status, expires_delta):
    """Put a claim lock on a task the way a dead worker leaves one behind."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = ?, claim_expires = ?, "
            "worker_pid = ? WHERE id = ?",
            (status, "ghost-host:29043", int(time.time()) + expires_delta,
             999999, task_id),
        )


def _row(conn, task_id):
    return conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid "
        "FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()


def test_expired_claim_on_a_ready_task_is_cleared(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="stuck", assignee="default")
        _stick_a_claim_on(conn, tid, status="ready", expires_delta=-3600)

        assert kb.release_orphaned_claims(conn) == 1
        row = _row(conn, tid)
        assert row["claim_lock"] is None
        assert row["claim_expires"] is None
        assert row["worker_pid"] is None
        assert row["status"] == "ready", (
            "the lock was the bug, not the phase - clearing a claim must not "
            "move the card to a different status"
        )
    finally:
        conn.close()


def test_unexpired_claim_is_left_alone(kanban_home):
    """An in-flight handoff is not an orphan."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="fresh", assignee="default")
        _stick_a_claim_on(conn, tid, status="ready", expires_delta=+3600)

        assert kb.release_orphaned_claims(conn) == 0
        assert _row(conn, tid)["claim_lock"] == "ghost-host:29043"
    finally:
        conn.close()


def test_running_tasks_are_left_to_release_stale_claims(kanban_home):
    """Do not duplicate the reclaimer's liveness logic.

    ``release_stale_claims`` checks whether the worker pid is alive and
    extends rather than reclaims when it is. Clearing a running task's lock
    here would spawn a duplicate beside a healthy worker.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="live", assignee="default")
        _stick_a_claim_on(conn, tid, status="running", expires_delta=-3600)

        assert kb.release_orphaned_claims(conn) == 0
        assert _row(conn, tid)["claim_lock"] == "ghost-host:29043"
    finally:
        conn.close()


def test_dispatcher_can_see_a_previously_deadlocked_card(kanban_home):
    """The end-to-end assertion: the card becomes dispatchable again.

    This one goes through ``dispatch_once``, so it fails if the helper is
    correct but never wired in.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="deadlocked", assignee="default")
        _stick_a_claim_on(conn, tid, status="ready", expires_delta=-3600)

        before = kb.dispatch_once(
            conn, dry_run=True, spawn_fn=lambda *a, **k: None,
        )
        assert before.orphaned_claims_released == 1, (
            "dispatch_once did not clear the orphaned claim - the helper is "
            "not wired into the tick"
        )
        assert tid in [t for t, _a, _w in before.spawned], (
            "the card is still invisible to the ready scan"
        )
    finally:
        conn.close()


def test_clearing_leaves_a_trace(kanban_home):
    """The failure mode was silence, so the repair must be visible."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="stuck", assignee="default")
        _stick_a_claim_on(conn, tid, status="ready", expires_delta=-3600)
        kb.release_orphaned_claims(conn)

        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            )
        ]
        assert "orphaned_claim_released" in kinds, (
            "no event was recorded - an operator reading `kanban tail` would "
            "again see a board go quiet with no reason given"
        )
    finally:
        conn.close()


def test_no_claims_is_a_cheap_no_op(kanban_home):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="clean", assignee="default")
        assert kb.release_orphaned_claims(conn) == 0
    finally:
        conn.close()
