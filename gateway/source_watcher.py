"""Notice when the gateway's own source changes on disk, and ask it to restart.

WHY THIS EXISTS
---------------
The gateway imports its modules once, at startup. Editing ``kanban_db.py`` --
or any other module -- changes nothing in the running process; the patch is
inert until somebody remembers to restart. That gap is invisible in exactly
the wrong way: the source says one thing and the running process does another,
so a session reads the fix, believes it is live, and reasons from behaviour
the gateway does not have.

WHAT THIS IS NOT
----------------
This is NOT hot module reloading. Reloading modules in place leaves a process
holding a mix of old and new objects, which is harder to reason about than
either version alone. This instead reuses the graceful restart that already
exists: ``request_restart()`` refuses new turns, waits for in-flight work to
finish, then exits cleanly. An external supervisor starts a fresh process with
the new code.

THE SAFETY PROPERTY
-------------------
Requesting a restart is only safe when something will bring the gateway BACK.
Under launchd (``KeepAlive: true``) or systemd a clean exit is restarted within
seconds; run by hand in a terminal, the same exit is just the gateway
disappearing mid-session because someone saved a file. So the watcher refuses
to arm unless an external supervisor is present -- see ``should_watch``. This
is the one decision in this module that must not be relaxed for convenience.

DEBOUNCE
--------
Editors write files in bursts: a `git checkout` of a branch rewrites hundreds
of files over several seconds, and a partially-applied tree is the worst
possible moment to restart into. The watcher therefore requires the
fingerprint to be STABLE for ``quiet_seconds`` before acting, so a burst
triggers one restart after it settles rather than one per file.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from gateway.restart import (
    EXTERNAL_GATEWAY_SUPERVISOR_ENV as EXTERNAL_SUPERVISOR_ENV,
    is_gateway_supervisor_process,
)

logger = logging.getLogger(__name__)

#: Opt-out for an operator who wants a supervised gateway to stay put.
DISABLE_ENV = "HERMES_GATEWAY_SOURCE_RELOAD_OFF"

#: Package directories whose contents define the running behaviour.
WATCHED_PACKAGES: tuple[str, ...] = ("gateway", "hermes_cli", "tools")

#: Seconds the tree must be unchanged before a restart is requested.
DEFAULT_QUIET_SECONDS = 5.0

#: Seconds between fingerprints.
DEFAULT_POLL_SECONDS = 5.0

_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "tests",
    }
)


def source_root() -> Path:
    """The checkout this module was imported from."""
    return Path(__file__).resolve().parent.parent


def iter_source_files(root: Path, packages: Sequence[str] = WATCHED_PACKAGES) -> Iterable[Path]:
    """Every ``.py`` file whose content defines the gateway's behaviour.

    Tests are excluded deliberately: editing a test cannot change what the
    running gateway does, and restarting a live process because someone
    touched a test file is pure cost.
    """
    for package in packages:
        base = root / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in _SKIP_DIR_NAMES for part in path.relative_to(root).parts):
                continue
            yield path


def fingerprint(root: Path, packages: Sequence[str] = WATCHED_PACKAGES) -> str:
    """A digest that changes iff a watched file's content changes.

    Content-hashed rather than mtime-stamped. A `git checkout` that moves away
    from a branch and straight back rewrites mtimes while restoring identical
    bytes; restarting for that would be a restart for nothing. Hashing also
    makes the check honest about the one thing it claims to detect.

    Unreadable files are folded in as a marker rather than skipped, so a file
    disappearing is itself a change.
    """
    digest = hashlib.sha256()
    for path in iter_source_files(root, packages):
        digest.update(str(path.relative_to(root)).encode("utf-8", "replace"))
        digest.update(b"\0")
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()


def should_watch(env: dict[str, str] | None = None) -> bool:
    """Whether arming the watcher is SAFE, not merely whether it is wanted.

    A restart request is an exit. Without a supervisor to start the next
    process, that is not a reload -- it is the gateway vanishing because
    somebody saved a file. Refusing here is what keeps this feature from
    turning an edit into an outage.

    Delegates the supervisor question to ``is_gateway_supervisor_process``
    rather than re-deciding it. That function is what the existing restart
    path already trusts to answer "will something bring me back", and a
    second, subtly different definition of the same predicate is how two
    guards end up disagreeing about the same command.
    """
    env = os.environ if env is None else env
    if str(env.get(DISABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return is_gateway_supervisor_process(env)


@dataclass
class SourceWatcher:
    """Fingerprint the tree on a tick; fire once when it settles on a change.

    Deliberately clock-injected and tick-driven rather than owning a sleep
    loop, so its behaviour is testable without waiting on real time.
    """

    root: Path
    on_change: Callable[[str], None]
    quiet_seconds: float = DEFAULT_QUIET_SECONDS
    packages: Sequence[str] = WATCHED_PACKAGES
    _baseline: str = field(default="", init=False)
    _pending: str = field(default="", init=False)
    _pending_since: float = field(default=0.0, init=False)
    _fired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._baseline = fingerprint(self.root, self.packages)

    def tick(self, now: float) -> bool:
        """Sample once. Returns True iff a restart was requested on this tick.

        Fires at most once per instance: after the request the process is on
        its way out, and a second request would be noise in the log of a
        gateway that is already draining.
        """
        if self._fired:
            return False

        current = fingerprint(self.root, self.packages)
        if current == self._baseline:
            # Reverted mid-burst (a checkout that came back to where it
            # started). Nothing to restart into.
            self._pending = ""
            return False

        if current != self._pending:
            self._pending = current
            self._pending_since = now
            logger.info(
                "Gateway source changed on disk; waiting %.0fs for the tree to "
                "settle before requesting a restart",
                self.quiet_seconds,
            )
            return False

        if (now - self._pending_since) < self.quiet_seconds:
            return False

        self._fired = True
        logger.info(
            "Gateway source has been stable for %.0fs since it changed; "
            "requesting a graceful restart so the new code is what runs",
            self.quiet_seconds,
        )
        try:
            self.on_change(current)
        except Exception:
            logger.exception("Source-reload restart request failed")
        return True
