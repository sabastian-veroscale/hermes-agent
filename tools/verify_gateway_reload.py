#!/usr/bin/env python3
"""Verify that a source edit reaches the RUNNING gateway without a manual restart.

This is an oracle, not a unit test. It asserts against the INSTALLED tree and
the LIVE process, because the defect it exists to catch is precisely "the fix
is on disk but not in the running process" -- a state every source-level check
reports as green.

Four checks, each with a named input that makes it fail:

  wired       the installed gateway/run.py arms the watcher from start()
              FAILS IF: the call site is deleted, or source_watcher.py is not
              vendored alongside it.

  supervised  the job that runs the gateway will start it again after a clean
              exit (launchd KeepAlive / systemd Restart), and passes
              --external-supervisor.
              FAILS IF: KeepAlive is set false, or the flag is dropped -- both
              of which turn a restart request into an outage.

  armed       should_watch() returns True for the environment that job supplies.
              FAILS IF: HERMES_GATEWAY_SOURCE_RELOAD_OFF is set in the job.

  current     the running process started AFTER the newest watched source file
              changed. This is the actual property under test: a gateway older
              than its own code is running code nobody can read.
              FAILS IF: you edit a watched file and nothing restarts within
              GRACE_SECONDS.

Exit 0 only when all four hold. `--json` prints machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

#: How long a change may sit unpicked-up before `current` calls it a failure.
#: Must exceed quiet period + poll interval + gateway startup, with headroom;
#: below that this check goes red every time somebody saves a file.
GRACE_SECONDS = 180.0

_TRUTHY = {"1", "true", "yes", "on"}


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def installed_root() -> Path:
    return hermes_home() / "hermes-agent"


# --------------------------------------------------------------------------
# check: wired
# --------------------------------------------------------------------------


def check_wired(root: Path) -> tuple[bool, str]:
    watcher = root / "gateway" / "source_watcher.py"
    run_py = root / "gateway" / "run.py"
    if not watcher.is_file():
        return False, f"{watcher} is not vendored — the deployed tree predates the watcher"
    if not run_py.is_file():
        return False, f"{run_py} missing"
    text = run_py.read_text(encoding="utf-8", errors="replace")
    if "def _start_source_watcher_task" not in text:
        return False, "gateway/run.py has no _start_source_watcher_task"
    # The call site, not just the definition: correct-but-never-invoked is the
    # exact failure mode this whole feature exists to prevent.
    body = _method_body(text, "    async def start(self)")
    if body is None:
        return False, "cannot locate GatewayRunner.start in gateway/run.py"
    if "self._start_source_watcher_task()" not in body:
        return False, "_start_source_watcher_task is defined but start() never calls it"
    return True, "installed gateway/run.py arms the watcher from start()"


def _method_body(text: str, header: str) -> str | None:
    """Text of one method, cut at the next sibling ``def`` at the same indent.

    A fixed character window would have worked when this was written and
    silently stopped covering the call site as the method grew -- ``start`` is
    already ~40 KB.
    """
    start = text.find(header)
    if start < 0:
        return None
    indent = len(header) - len(header.lstrip())
    sibling = " " * indent + "def "
    async_sibling = " " * indent + "async def "
    lines = text[start:].splitlines(keepends=True)
    out = [lines[0]]
    for line in lines[1:]:
        if line.startswith(sibling) or line.startswith(async_sibling):
            break
        out.append(line)
    return "".join(out)


# --------------------------------------------------------------------------
# check: supervised
# --------------------------------------------------------------------------


def _launchd_job(root: Path) -> tuple[dict | None, Path | None]:
    """The launchd job for THIS HERMES_HOME.

    There is more than one gateway job on a machine running profiles (the
    ``-calcifer`` profile has its own HERMES_HOME), and picking the first
    plist that merely looks like a gateway reports on somebody else's
    process. Matched on HERMES_HOME, which is what actually decides which
    tree a job runs.
    """
    home = hermes_home().resolve()
    fallback: tuple[dict, Path] | None = None
    for candidate in sorted((Path.home() / "Library" / "LaunchAgents").glob("*hermes*gateway*.plist")):
        try:
            data = plistlib.loads(candidate.read_bytes())
        except Exception:
            continue
        args = [str(a) for a in data.get("ProgramArguments", [])]
        if "gateway" not in args or "run" not in args:
            continue
        job_home = (data.get("EnvironmentVariables") or {}).get("HERMES_HOME")
        if job_home and Path(str(job_home)).resolve() == home:
            return data, candidate
        if fallback is None and not job_home:
            # A job that does not pin HERMES_HOME inherits the default one.
            fallback = (data, candidate)
    if fallback is not None:
        return fallback
    return None, None


def _systemd_unit() -> tuple[str, str] | None:
    for unit in ("hermes-gateway.service", "hermes.service"):
        try:
            out = subprocess.run(
                ["systemctl", "--user", "cat", unit],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0 and out.stdout.strip():
            return unit, out.stdout
    return None


def check_supervised(root: Path) -> tuple[bool, str, dict[str, str]]:
    """Returns (ok, detail, env-the-job-supplies)."""
    plist, path = _launchd_job(root)
    if plist is not None:
        env = {str(k): str(v) for k, v in (plist.get("EnvironmentVariables") or {}).items()}
        args = [str(a) for a in plist.get("ProgramArguments", [])]
        if "--external-supervisor" in args:
            env["HERMES_GATEWAY_EXTERNAL_SUPERVISOR"] = "1"
        keep = plist.get("KeepAlive")
        alive = keep is True or (isinstance(keep, dict) and keep)
        if not alive:
            return False, f"{path} has KeepAlive={keep!r} — a clean exit would not come back", env
        if "--external-supervisor" not in args:
            return (
                False,
                f"{path} does not pass --external-supervisor — the gateway will not "
                "know a supervisor owns it",
                env,
            )
        return True, f"launchd {plist.get('Label')} restarts it and passes the flag", env

    found = _systemd_unit()
    if found is not None:
        unit, text = found
        restart = ""
        for line in text.splitlines():
            if line.strip().lower().startswith("restart="):
                restart = line.split("=", 1)[1].strip().lower()
        env = {"INVOCATION_ID": "systemd"}
        if restart not in {"always", "on-failure", "on-abnormal", "on-success"}:
            return False, f"systemd {unit} has Restart={restart or 'no'}", env
        if "--external-supervisor" in text:
            env["HERMES_GATEWAY_EXTERNAL_SUPERVISOR"] = "1"
        return True, f"systemd {unit} Restart={restart}", env

    return False, "no launchd job or systemd unit found that runs the gateway", {}


# --------------------------------------------------------------------------
# check: armed
# --------------------------------------------------------------------------


def check_armed(root: Path, job_env: dict[str, str]) -> tuple[bool, str]:
    sys.path.insert(0, str(root))
    try:
        from gateway.source_watcher import DISABLE_ENV, should_watch
    except Exception as exc:  # pragma: no cover - import failure is the message
        return False, f"cannot import the installed watcher: {exc}"
    finally:
        sys.path.pop(0)

    if str(job_env.get(DISABLE_ENV, "")).strip().lower() in _TRUTHY:
        return False, f"{DISABLE_ENV} is set in the job environment — watching is off"
    if not should_watch(job_env):
        return False, "should_watch() says no for the environment the job supplies"
    return True, "should_watch() arms under the job's environment"


# --------------------------------------------------------------------------
# check: current
# --------------------------------------------------------------------------


def _gateway_start_time(root: Path) -> float | None:
    """Wall-clock start of the running gateway, from its own heartbeat."""
    beat = hermes_home() / "state" / "gateway.heartbeat"
    try:
        data = json.loads(beat.read_text())
    except Exception:
        return None
    pid = data.get("pid")
    started = data.get("start_time")
    if not isinstance(started, (int, float)) or not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)  # signal 0: liveness only, delivers nothing
    except OSError:
        return None
    return float(started)


def check_current(root: Path) -> tuple[bool, str]:
    if root != installed_root():
        # The heartbeat describes the INSTALLED gateway. Comparing it against
        # some other checkout's mtimes measures nothing, and would report a
        # branch under active edit as a broken deployment.
        return True, f"skipped — {root} is not the installed tree"

    sys.path.insert(0, str(root))
    try:
        from gateway.source_watcher import iter_source_files
    except Exception as exc:
        return False, f"cannot import the installed watcher: {exc}"
    finally:
        sys.path.pop(0)

    started = _gateway_start_time(root)
    if started is None:
        return False, "no live gateway found (stale or absent heartbeat)"

    newest = 0.0
    newest_path = ""
    for path in iter_source_files(root):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest:
            newest, newest_path = mtime, str(path.relative_to(root))

    lag = newest - started
    if lag <= 0:
        return True, f"running process is newer than every watched file (by {-lag:.0f}s)"
    if lag < GRACE_SECONDS:
        return True, f"{newest_path} changed {lag:.0f}s ago; restart still within grace"
    return (
        False,
        f"{newest_path} changed {lag / 60:.1f} min after the running process "
        f"started and no restart followed — the gateway is running code that is "
        f"not what is on disk",
    )


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--root",
        default=None,
        help="gateway checkout to verify (default: the INSTALLED tree)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else installed_root()
    results: list[tuple[str, bool, str]] = []

    ok, detail = check_wired(root)
    results.append(("wired", ok, detail))

    sup_ok, sup_detail, job_env = check_supervised(root)
    results.append(("supervised", sup_ok, sup_detail))

    ok, detail = check_armed(root, job_env)
    results.append(("armed", ok, detail))

    ok, detail = check_current(root)
    results.append(("current", ok, detail))

    failed = [name for name, ok, _ in results if not ok]

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "ok": not failed,
                    "checks": [
                        {"name": n, "ok": o, "detail": d} for n, o, d in results
                    ],
                },
                indent=2,
            )
        )
    else:
        for name, ok, detail in results:
            print(f"{'OK  ' if ok else 'FAIL'}  {name:<11} {detail}")

    if failed:
        # The oracle tails the last line, so make it the verdict.
        if not args.json:
            print(f"gateway source reload NOT proven: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
