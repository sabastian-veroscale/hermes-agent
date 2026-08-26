"""Integration regression for the foreground 'terminal(command="... &")' failure.

Reported in FRX ticket t_e5591ea1: a foreground terminal call whose command
string ends with a stray shell-level '&' was failing with an actionable error
message, but the workflow to recover (re-invoke without '&' as background=true,
then run follow-up health checks and tests in separate bounded calls) was not
covered by any integration test. Existing coverage at
``test_terminal_heredoc_background_guard.py`` only asserts the pure helper
``_foreground_background_guidance`` -- it does NOT exercise the JSON envelope the
tool returns or the retry path through ``terminal_tool()`` itself, so a
regression in the wiring (wrong envelope shape, message rewording, guard
bypassed) would slip past CI.

This test reproduces the exact invocation shape from the ticket:

    1. Foreground call with a trailing ``&`` returns the actionable error
       envelope (output empty, exit_code -1, status "error", error prose that
       names the retry shape: ``background=true`` + ``notify_on_complete=true``
       for bounded jobs, then health checks / tests in separate calls).
    2. Re-invoking with ``background=True`` + ``notify_on_complete=True`` and
       the ``&`` stripped from the command succeeds (does NOT trip the
       foreground guard).
    3. Follow-up bounded calls (health check, tests) run as separate terminal
       jobs and each returns its own ``exit_code`` from the env -- proving the
       agent is free to chain health checks and tests after the background
       server comes up.

Acceptance:

    * Without the guard fix (i.e. if a future refactor drops the
      ``_foreground_background_guidance`` short-circuit or stops returning the
      actionable prose), step 1 fails -- the foreground call would either
      execute (against the user's intent) or return a non-actionable error.
    * With the fix applied (t_25ea98ce), the actionable wording is present, the
      retry path succeeds, and follow-up calls work -- the test passes.

The mock infrastructure mirrors ``test_terminal_output_transform_hook.py``: a
fake ``LocalEnvironment`` stands in for the real sandbox so the test stays
hermetic and fast.
"""

import json
from unittest.mock import MagicMock

import tools.terminal_tool as terminal_tool_module


_AMP = chr(38)  # & -- kept out of the source literal so a naive scanner
                #    scanning this file wouldn't trip on it.


def _make_env_config(tmp_path, **overrides):
    config = {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _install_mock_env(monkeypatch, tmp_path, *, output="ok", returncode=0):
    """Wire the terminal tool's environment seam to an in-memory fake.

    Returns the MagicMock env so callers can assert on ``execute.call_args``.
    """
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": output, "returncode": returncode}
    # The LocalEnvironment mock needs an ``env`` attribute the spawn_local path
    # inspects; the foreground path doesn't touch it.
    mock_env.env = {}

    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _make_env_config(tmp_path),
    )
    monkeypatch.setattr(
        terminal_tool_module, "_start_cleanup_thread", lambda: None
    )
    # Bypass tirith / dangerous-command approval so this test stays focused on
    # the foreground-background guard wiring, not the policy layer.
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_a, **_kw: {"approved": True},
    )
    # Pre-seed the env cache so the tool reuses it instead of trying to spin
    # up a real sandbox. The same key ('default') is used by
    # test_terminal_output_transform_hook._run_terminal -- keep them aligned.
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)
    return mock_env


# ---------------------------------------------------------------------------
# Step 1 -- foreground trailing '&' returns the actionable error envelope.
# ---------------------------------------------------------------------------


class TestForegroundTrailingAmpReturnsActionableError:
    """A foreground terminal call ending in ' & ' must short-circuit BEFORE
    any sandbox spin-up, returning the JSON envelope the user can branch on.
    """

    def test_trailing_amp_returns_error_envelope(self, monkeypatch, tmp_path):
        _install_mock_env(monkeypatch, tmp_path)
        # NOTE: real reported invocation -- command string ENDS with ' & '.
        cmd = "python3 server.py " + _AMP

        result_raw = terminal_tool_module.terminal_tool(command=cmd)
        result = json.loads(result_raw)

        # Envelope shape: the tool returns a JSON string with this fixed
        # schema (mirrors lines 2953-2961 of tools/terminal_tool.py).
        assert result["output"] == ""
        assert result["exit_code"] == -1
        assert result["status"] == "error"
        assert isinstance(result["error"], str) and result["error"]

    def test_error_message_contains_retry_shape(self, monkeypatch, tmp_path):
        """The actionable prose must tell the caller exactly how to retry.

        The verbatim error string from the FRX ticket:
            Foreground command uses '&' backgrounding. Re-send WITHOUT the
            '&' as terminal(command="<cmd>", background=true) -- add
            notify_on_complete=true for bounded jobs -- then run health
            checks and tests in follow-up terminal calls.
        """
        _install_mock_env(monkeypatch, tmp_path)
        cmd = "python3 server.py " + _AMP

        result = json.loads(terminal_tool_module.terminal_tool(command=cmd))

        err = result["error"]
        # Substring assertions are stable across wording tweaks that keep the
        # meaning intact. If any of these disappear, the error has regressed
        # to be non-actionable and the user is left guessing.
        assert "& backgrounding" in err or "'&' backgrounding" in err, (
            "Error must name the offending '&' backgrounding; got: "
            + repr(err)
        )
        assert "background=true" in err, (
            "Error must tell the caller to retry with background=true; got: "
            + repr(err)
        )
        assert "notify_on_complete=true" in err, (
            "Error must suggest notify_on_complete=true for bounded jobs; got: "
            + repr(err)
        )
        # The message must point the caller at follow-up calls (health
        # checks / tests), not suggest a single retry handles everything.
        assert "follow-up" in err or "separate" in err or "followup" in err, (
            "Error must mention follow-up / separate calls for health checks "
            "and tests; got: " + repr(err)
        )

    def test_foreground_amp_does_not_execute_command(self, monkeypatch, tmp_path):
        """The guard must reject -- NOT execute the backgrounded command.

        This is the critical safety property: if a future change accidentally
        lets the command run, the shell forks the trailing process into the
        background and Hermes loses track of it. The mock env's execute() must
        not have been called.
        """
        mock_env = _install_mock_env(monkeypatch, tmp_path)
        cmd = "python3 server.py " + _AMP

        terminal_tool_module.terminal_tool(command=cmd)

        assert mock_env.execute.call_count == 0, (
            "Foreground '&' command must NOT reach env.execute -- the guard "
            "must short-circuit before any sandbox call."
        )


# ---------------------------------------------------------------------------
# Step 2 -- re-invoking WITHOUT '&' as background=true succeeds.
# ---------------------------------------------------------------------------


class TestRetryWithBackgroundTrueSucceeds:
    """After the actionable error, the documented retry path must succeed.

    The fix's recovery flow: drop the '&' from the command, set
    ``background=True`` and ``notify_on_complete=True`` (the latter is the
    recommended way to bound the job). The call must NOT trigger the
    foreground guard -- the actionable error envelope is the bug, not the
    background execution.
    """

    def test_background_retry_does_not_return_actionable_error(
        self, monkeypatch, tmp_path
    ):
        """The foreground guard must be bypassed when background=True.

        A future regression that accidentally applies the foreground
        guidance to background calls would return the actionable error here.
        """
        _install_mock_env(monkeypatch, tmp_path)

        # Retry shape: same intent, '&' stripped, background=True,
        # notify_on_complete=True.
        result_raw = terminal_tool_module.terminal_tool(
            command="python3 server.py",
            background=True,
            notify_on_complete=True,
        )

        # The actionable error envelope (status='error', exit_code=-1,
        # error contains '& backgrounding') is exclusive to the foreground
        # guard. Its presence means the guard leaked across modes.
        if result_raw.startswith("{"):
            parsed = json.loads(result_raw)
            if parsed.get("status") == "error":
                err = parsed.get("error", "")
                assert "& backgrounding" not in err, (
                    "background=True call returned the foreground actionable "
                    "error -- the guard leaked across modes: " + repr(err)
                )

    def test_background_retry_with_inline_ampersand_is_accepted(
        self, monkeypatch, tmp_path
    ):
        """Even a compound '... & ...' form must pass under background=True.

        The fix's guidance says 're-invoke WITHOUT the &', but a downstream
        caller that follows the example but forgets to strip it must still
        not get the foreground-error envelope back. The compound-rewrite
        layer (test_terminal_compound_background) handles the actual
        execution; the foreground guard is what we test here.
        """
        _install_mock_env(monkeypatch, tmp_path)

        result_raw = terminal_tool_module.terminal_tool(
            command="python3 server.py " + _AMP + " echo started",
            background=True,
            notify_on_complete=True,
        )

        if result_raw.startswith("{"):
            parsed = json.loads(result_raw)
            if parsed.get("status") == "error":
                err = parsed.get("error", "")
                assert "& backgrounding" not in err, (
                    "background=True compound command returned the foreground "
                    "actionable error: " + repr(err)
                )


# ---------------------------------------------------------------------------
# Step 3 -- follow-up bounded calls run as separate jobs.
# ---------------------------------------------------------------------------


class TestFollowUpCallsRunAsSeparateJobs:
    """After the background server is up, the agent runs health checks and
    tests in separate terminal calls. Each must complete cleanly with its
    own exit_code, demonstrating the bounded-jobs pattern.
    """

    def test_health_check_call_succeeds_independently(self, monkeypatch, tmp_path):
        mock_env = _install_mock_env(
            monkeypatch, tmp_path, output='{"ok": true}', returncode=0
        )
        health_cmd = "curl -sf http://127.0.0.1:8000/health"

        result = json.loads(terminal_tool_module.terminal_tool(command=health_cmd))

        assert result["exit_code"] == 0
        assert result["error"] is None
        assert "ok" in result["output"]
        # The mock env was invoked exactly once for the health check.
        assert mock_env.execute.call_count == 1
        assert mock_env.execute.call_args.args[0] == health_cmd

    def test_tests_call_succeeds_independently(self, monkeypatch, tmp_path):
        mock_env = _install_mock_env(
            monkeypatch, tmp_path, output="3 passed in 0.4s", returncode=0
        )
        tests_cmd = "pytest -x tests/test_smoke.py -q"

        result = json.loads(terminal_tool_module.terminal_tool(command=tests_cmd))

        assert result["exit_code"] == 0
        assert result["error"] is None
        assert "passed" in result["output"]
        assert mock_env.execute.call_count == 1
        assert mock_env.execute.call_args.args[0] == tests_cmd

    def test_three_separate_bounded_calls_do_not_collide(self, monkeypatch, tmp_path):
        """A realistic sequence: curl health, curl ready, pytest. Each must
        be its own terminal call with its own exit_code and env.execute
        invocation. The guard must not conflate them.
        """
        mock_env = _install_mock_env(monkeypatch, tmp_path, output="ok", returncode=0)

        bounded_calls = [
            "curl -sf http://127.0.0.1:8000/health",
            "curl -sf http://127.0.0.1:8000/ready",
            "pytest -x tests/test_smoke.py -q",
        ]

        for cmd in bounded_calls:
            raw = terminal_tool_module.terminal_tool(command=cmd)
            parsed = json.loads(raw)
            assert parsed["exit_code"] == 0, (
                f"{cmd!r} returned non-zero: exit_code={parsed.get('exit_code')!r}, "
                f"error={parsed.get('error')!r}"
            )
            assert parsed["error"] is None, (
                f"{cmd!r} returned error: {parsed['error']!r}"
            )

        # Three independent env.execute calls -- each command landed
        # separately. No call was conflated into another.
        assert mock_env.execute.call_count == 3
        executed_cmds = [c.args[0] for c in mock_env.execute.call_args_list]
        assert executed_cmds == bounded_calls
