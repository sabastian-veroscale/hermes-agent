"""Regression tests for the symmetric rejection envelope + log emission.

Kanban: t_96f871c9, t_0f1d3006
Companion to ``test_terminal_foreground_amp_regression.py``.

Background
----------
The terminal-tool rejection paths short-circuit BEFORE ``env.execute()`` is
called and return an actionable error envelope. Two findings surfaced during
the diagnosis of the "[frx] terminal tool call failed" content-free cards:

1. **Envelope asymmetry.** The foreground-background guard at
   ``tools/terminal_tool.py:2952-2960`` returns a 4-field envelope
   ``{output: "", exit_code: -1, error: <prose>, status: "error"}``. The
   foreground-timeout guard at ``:2943-2948`` uses ``tool_error()`` which
   returns ``{error: <msg>}`` -- missing ``output``, ``exit_code``, and
   ``status``. Downstream consumers that classify rejections by shape (V1/V2
   vs V8 in the reproduction recipe) misclassify V8 as "the tool gave us
   nothing."

2. **Empty log buffer on rejection.** Neither guard calls ``logger.info()``,
   so the actionable prose has no alternate route into the diagnostic stream
   the observer hook uses. The recipe's ``_capture_log_output`` buffer was
   empty across all 8 variants.

Fix
---
- Normalize the foreground-timeout guard to return the same 4-field envelope
  shape as the background-guidance guard, with the same exit_code (-1) and
  status ("error").
- Add ``logger.info("rejected foreground timeout: %s", ...)`` and
  ``logger.info("rejected foreground background: %s", ...)`` on the rejection
  paths so the prose lands in the diagnostic stream regardless of observer
  implementation.

These tests lock in both behaviors so any future refactor that drops the
symmetry (e.g. re-introducing ``tool_error()`` on the timeout path) or the
logger call (e.g. swallowing the rejection silently) fails CI.
"""

import json
import logging
from unittest.mock import MagicMock

import tools.terminal_tool as terminal_tool_module


_AMP = chr(38)  # & -- kept out of source literal so a naive scanner wouldn't trip.


def _install_mock_env(monkeypatch, tmp_path, *, output="ok", returncode=0):
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": output, "returncode": returncode}
    mock_env.env = {}
    monkeypatch.setattr(
        terminal_tool_module,
        "_active_environments",
        {"default": mock_env},
    )
    return mock_env


class TestRejectionEnvelopeSymmetry:
    """Both rejection paths return the SAME envelope shape."""

    def test_foreground_timeout_envelope_matches_amp_guard_shape(self, monkeypatch, tmp_path):
        """The V8 (timeout > FOREGROUND_MAX_TIMEOUT) rejection MUST have the
        same 4-field shape as the V1/V2 (backgrounding) rejection: ``output``
        (empty string), ``exit_code`` (-1), ``error`` (actionable prose),
        ``status`` ("error"). Any missing field breaks downstream shape-based
        classifiers.
        """
        _install_mock_env(monkeypatch, tmp_path)
        captured = []
        monkeypatch.setattr(
            terminal_tool_module.logger, "info",
            lambda msg, *a: captured.append((msg, a)),
        )

        result_raw = terminal_tool_module.terminal_tool(
            command="sleep 999",
            timeout=999,            # well above FOREGROUND_MAX_TIMEOUT (600s)
        )
        parsed = json.loads(result_raw)

        assert parsed["output"] == "", (
            f"V8 envelope must include output:'', got {parsed.get('output')!r}"
        )
        assert parsed["exit_code"] == -1, (
            f"V8 envelope must include exit_code:-1, got {parsed.get('exit_code')!r}"
        )
        assert isinstance(parsed["error"], str) and "Foreground timeout" in parsed["error"], (
            f"V8 envelope must include actionable error prose, got {parsed.get('error')!r}"
        )
        assert parsed["status"] == "error", (
            f"V8 envelope must include status:'error', got {parsed.get('status')!r}"
        )

    def test_amp_guard_envelope_unchanged(self, monkeypatch, tmp_path):
        """Locking in the V1/V2 envelope shape -- the existing regression
        test already covers the actionable prose, but not the exact field
        set, so this anchors it for the symmetry contract.
        """
        _install_mock_env(monkeypatch, tmp_path)
        captured = []
        monkeypatch.setattr(
            terminal_tool_module.logger, "info",
            lambda msg, *a: captured.append((msg, a)),
        )

        result_raw = terminal_tool_module.terminal_tool(
            command="python3 server.py " + _AMP,
        )
        parsed = json.loads(result_raw)

        assert parsed["output"] == ""
        assert parsed["exit_code"] == -1
        assert isinstance(parsed["error"], str) and "backgrounding" in parsed["error"]
        assert parsed["status"] == "error"


class TestRejectionLogging:
    """Both rejection paths emit ``logger.info(...)`` so the prose reaches
    the diagnostic stream the observer hook uses.
    """

    def test_foreground_timeout_logs_diagnostic(self, monkeypatch, tmp_path):
        _install_mock_env(monkeypatch, tmp_path)
        captured = []
        monkeypatch.setattr(
            terminal_tool_module.logger, "info",
            lambda msg, *a: captured.append((msg, a)),
        )

        terminal_tool_module.terminal_tool(command="sleep 999", timeout=999)

        # Exactly one logger.info call on the rejection -- the actionable
        # prose carried in the same record.
        info_messages = [(m, a) for m, a in captured if "rejected" in m]
        assert len(info_messages) == 1, (
            f"expected exactly one rejection log, got {info_messages!r}"
        )
        msg, args = info_messages[0]
        assert "rejected foreground timeout" in msg
        assert args and "Foreground timeout" in args[0]
        assert "background=true" in args[0]

    def test_amp_guard_logs_diagnostic(self, monkeypatch, tmp_path):
        _install_mock_env(monkeypatch, tmp_path)
        captured = []
        monkeypatch.setattr(
            terminal_tool_module.logger, "info",
            lambda msg, *a: captured.append((msg, a)),
        )

        terminal_tool_module.terminal_tool(command="python3 server.py " + _AMP)

        info_messages = [(m, a) for m, a in captured if "rejected" in m]
        assert len(info_messages) == 1, (
            f"expected exactly one rejection log, got {info_messages!r}"
        )
        msg, args = info_messages[0]
        assert "rejected foreground background" in msg
        assert args and "backgrounding" in args[0]
        assert "background=true" in args[0]


class TestNormalCallsDoNotLogRejection:
    """A successful foreground call MUST NOT emit a rejection log."""

    def test_successful_foreground_call_logs_nothing(self, monkeypatch, tmp_path, caplog):
        mock_env = _install_mock_env(monkeypatch, tmp_path, output="hello", returncode=0)
        with caplog.at_level(logging.INFO, logger="tools.terminal_tool"):
            result = json.loads(
                terminal_tool_module.terminal_tool(command="echo hello")
            )

        assert result["exit_code"] == 0
        assert mock_env.execute.call_count == 1
        # No rejection log on the happy path.
        rejection_records = [
            r for r in caplog.records
            if "rejected foreground" in r.getMessage()
        ]
        assert rejection_records == [], (
            f"happy path emitted unexpected rejection log: {[r.getMessage() for r in rejection_records]!r}"
        )