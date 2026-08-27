"""Every worker is told to write its work order before it starts working.

WHY THIS FILE EXISTS
--------------------
A worker has a bounded iteration budget. When it runs out the process is
killed and everything it learned dies with it, so the retry re-derives the
same findings against the same budget and dies the same way. Measured on the
ops board over the 7 days to 2026-08-27: 148 runs ended in "iteration budget
exhausted" (93 timed_out + 55 gave_up) — the largest failure class after
crash-misclassification.

The carrier is the task workspace, which is keyed by task id and reused across
retries. The instruction lives in ``build_worker_context`` rather than in card
bodies on purpose: editing 214 open card bodies fixes those 214 and none of the
next 214, and leaves two copies of the protocol to drift apart.

FALSIFIER
---------
Delete the ``_write_first_block`` call from ``build_worker_context`` and
``test_protocol_is_in_every_worker_context`` goes RED. Move the call below the
``## Body`` append and ``test_protocol_precedes_the_body`` goes RED.
``test_no_workspace_means_no_promise`` is the one that must NOT be satisfiable
by simply always emitting the block.
"""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

WORKSPACE = "/tmp/kanban-ws/t_test0001"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _context(workspace=WORKSPACE, body="do the thing"):
    """Render a real worker context for a real task row."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Test card", body=body)
        if workspace:
            kb.set_workspace_path(conn, task_id, workspace)
        return kb.build_worker_context(conn, task_id)
    finally:
        conn.close()


def test_protocol_is_in_every_worker_context(kanban_home):
    ctx = _context()
    assert "Write-first protocol" in ctx
    assert f"{WORKSPACE}/{kb.WORKER_NOTES_FILENAME}" in ctx, (
        "the worker was not told the absolute path to write to - a relative "
        "instruction lands wherever cwd happens to be"
    )


def test_protocol_precedes_the_body(kanban_home):
    """Order is the whole point.

    Placed after the body, a worker reads its task and starts working; the
    instruction to write first arrives after it has already not done so.
    """
    ctx = _context()
    assert ctx.index("Write-first protocol") < ctx.index("## Body"), (
        "the write-first block must come BEFORE the task body"
    )


def test_it_tells_the_retry_to_read_before_re_deriving(kanban_home):
    """Half the mechanism is writing; the other half is reading."""
    ctx = _context()
    head = ctx[: ctx.index("## Body")]
    assert "before anything else" in head
    assert "do not re-derive" in head


@pytest.mark.parametrize("empty", [None, ""])
def test_no_workspace_means_no_promise(kanban_home, empty):
    """Do not instruct a worker to write to a path that will not persist.

    This is the assertion that stops the block from being unconditional
    boilerplate: with nowhere durable to write, the honest output is silence,
    not an instruction the worker cannot honour.
    """
    ctx = _context(workspace=empty)
    assert "Write-first protocol" not in ctx


def test_block_builder_is_pure_and_path_shaped():
    joined = "\n".join(kb._write_first_block("/tmp/ws/"))
    assert "/tmp/ws/WORK-ORDER.md" in joined
    assert "//WORK-ORDER.md" not in joined, (
        "trailing slash produced a doubled separator"
    )


def test_it_does_not_swamp_the_task():
    """Context costs iterations. The cure must not be the disease."""
    block = "\n".join(kb._write_first_block(WORKSPACE))
    assert len(block) < 2000, (
        f"write-first block is {len(block)} chars - it is spending the budget "
        "it exists to save"
    )
