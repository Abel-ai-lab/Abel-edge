"""Prevent joblib/fork hangs in CLI invocations.

Problem history
---------------
2026-04-17: daily cron hung 2.4 hours silently. joblib loky workers
  forked from multi-threaded Python inherited locked mutexes from
  threads that did not exist in the child; workers deadlocked in library
  init; parent stuck in wait_result_broken_or_wakeup forever.
2026-04-18: manual ``abel-edge paper`` hung 5h13m with the same
  signature. The 2026-04-17 mitigation was in the ``run_daily.sh``
  wrapper (LOKY_MAX_CPU_COUNT cap, timeout, trap). Bare CLI invocations
  — interactive debugging, agent-driven runs, ad-hoc operator commands —
  had none of those safeguards, so the class bug recurred.

Root cause
----------
Linux fork() from a multi-threaded Python process is fragile. Any thread
that ran (even briefly — numpy/OMP init creates threads on import) leaves
locked mutex state in the address space. fork() copies the memory but not
the threads, so the child inherits locked mutexes whose owners no longer
exist. The child deadlocks on the first library init that tries to
acquire them.

Three-layer fix (applied at CLI entry, before heavy imports)
------------------------------------------------------------
Layer 1  Env vars cap concurrency: LOKY_MAX_CPU_COUNT=4,
         OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2. Fewer forks and fewer
         threads per fork = smaller surface for the deadlock race.
         Does not require forkserver support anywhere else.
Layer 2  multiprocessing start method = forkserver. Workers spawn from
         a pristine helper process that never imported threaded code —
         no locked mutexes to inherit. This is the actual cure on
         Linux/macOS. Windows already uses spawn, which is immune.
Layer 3  SIGTERM/SIGINT/atexit trap recursively kills descendants via
         /proc walk. Backstop: if a worker somehow survives parent exit
         (e.g. holding an inherited fd 200 flock, as in 2026-04-17), the
         trap reaps the whole tree before the CLI returns — so the next
         cron invocation does not find the lock held by a zombie.

Optional Layer 4 (install_global_timeout): SIGALRM wall-clock cap that
fires Layer-3 kill and exits 124 on expiry. Opt-in because a mid-run
kill changes the semantics of the command; callers must explicitly
request it.

Layer ordering matters
----------------------
Env vars MUST be set before numpy/sklearn/joblib import (OMP reads them
on first thread creation). mp start method MUST be set before any
multiprocessing activity. Hence ``apply()`` is called as the very first
statement of ``abel_edge/cli.py`` — before even ``click`` is imported.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from typing import Final

_ENV_DEFAULTS: Final[dict[str, str]] = {
    "LOKY_MAX_CPU_COUNT": "4",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}

_TIMEOUT_EXIT_CODE: Final[int] = 124  # POSIX convention (same as /usr/bin/timeout)
# signal.alarm() takes a C unsigned int. Cap well below 2**32 - 1 so the
# alarm() call cannot OverflowError. One week is a deliberately generous
# upper bound — far longer than any sensible CLI invocation, but small
# enough to be a clearly bounded value.
_MAX_TIMEOUT_SECONDS: Final[int] = 7 * 24 * 60 * 60

_applied = False
_trap_installed = False
_alarm_installed = False


def apply() -> None:
    """Apply Layer 1 (env vars) and Layer 2 (forkserver). Idempotent.

    MUST be called before any import that triggers numpy/sklearn/joblib
    loading, because OMP and MKL only read their thread-count env vars
    once — on the first thread creation. Later writes are ignored.
    """
    global _applied
    if _applied:
        return
    _applied = True

    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    if sys.platform == "win32":
        return  # spawn is the default; no fork deadlock to prevent
    try:
        import multiprocessing as mp

        # force=False so we do not override a prior explicit choice
        mp.set_start_method("forkserver", force=False)
    except (RuntimeError, ImportError, ValueError):
        # RuntimeError: start method already set — honor the prior setter.
        # ImportError: multiprocessing unavailable on this build.
        # ValueError: this POSIX build does not ship the forkserver method
        #   (rare embedded/stripped Python). Falling back to the default
        #   start method is correct — better to lose Layer 2 than to crash
        #   every CLI invocation at startup.
        pass


def _descendant_pids(root_pid: int) -> list[int]:
    """Return PIDs of every descendant of ``root_pid`` via /proc walk.

    Linux-only (other platforms return []). No subprocess spawn, no psutil
    dependency. Snapshots /proc once, so a descendant born after the scan
    will not appear — caller should re-scan after a grace period to catch
    late arrivals.
    """
    try:
        entries = list(os.scandir("/proc"))
    except (FileNotFoundError, PermissionError):
        return []

    pid_to_children: dict[int, list[int]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat") as fh:
                text = fh.read()
        except OSError:
            continue

        # /proc/<pid>/stat: "pid (comm) state ppid ..." where comm can
        # contain spaces and parens. Parse after the rightmost ')'.
        close_paren = text.rfind(")")
        if close_paren < 0:
            continue
        fields = text[close_paren + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
            pid = int(entry.name)
        except ValueError:
            continue
        pid_to_children.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in pid_to_children.get(pid, []):
            descendants.append(child)
            stack.append(child)
    return descendants


def _kill_descendants(grace_seconds: float = 0.5) -> int:
    """Recursively SIGTERM then SIGKILL all descendants. Return count reaped."""
    our_pid = os.getpid()
    targets = _descendant_pids(our_pid)
    if not targets:
        return 0

    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    if grace_seconds > 0:
        time.sleep(grace_seconds)

    # Re-scan: a worker's own children may have been spawned between our
    # first scan and the SIGTERM, or between SIGTERM and now.
    for pid in _descendant_pids(our_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    return len(targets)


def install_tree_kill_trap() -> None:
    """Recursively kill descendants on SIGTERM/SIGINT and at exit. Idempotent.

    Call at the start of long-running CLI commands (paper, run, dashboard)
    so that any joblib worker that outlives deadlock detection is reaped
    before the CLI returns. Prevents the 2026-04-17 flock-zombie incident
    where fd 200 was held by an orphan for 6 days.
    """
    global _trap_installed
    if _trap_installed:
        return
    _trap_installed = True

    def _handler(signum: int, _frame) -> None:
        sys.stderr.write(f"[runtime_harden] signal {signum} — killing descendants\n")
        sys.stderr.flush()
        _kill_descendants()
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Non-main thread or signal not supported on this platform
            pass

    atexit.register(_kill_descendants)


def install_global_timeout(seconds: int) -> None:
    """Install a wall-clock timeout. On expiry, kill tree and exit 124.

    Opt-in. Pass 0 (or negative) to disable. No-op on Windows (no SIGALRM).
    Values above ``_MAX_TIMEOUT_SECONDS`` (one week) are clamped down with a
    stderr warning — protects against ``signal.alarm()`` ``OverflowError``
    on enormous inputs (ABEL_EDGE_TIMEOUT_SECONDS=99999999999999) which
    would otherwise crash every CLI invocation at startup.
    """
    global _alarm_installed
    if _alarm_installed or seconds <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    if seconds > _MAX_TIMEOUT_SECONDS:
        sys.stderr.write(
            f"[runtime_harden] requested timeout {seconds}s exceeds cap "
            f"{_MAX_TIMEOUT_SECONDS}s; clamping to cap\n"
        )
        seconds = _MAX_TIMEOUT_SECONDS

    def _handler(_signum: int, _frame) -> None:
        sys.stderr.write(f"[runtime_harden] TIMEOUT after {seconds}s — killing descendants\n")
        sys.stderr.flush()
        _kill_descendants()
        os._exit(_TIMEOUT_EXIT_CODE)

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
    except (OverflowError, OSError, ValueError) as exc:
        # Belt-and-suspenders: cap above should prevent OverflowError, but if
        # something else rejects the value (very small embedded systems, weird
        # signal state), do not crash the CLI.
        sys.stderr.write(f"[runtime_harden] could not install timeout: {exc}\n")
        return
    _alarm_installed = True


def protect_cli_command(timeout_seconds: int = 0) -> None:
    """Convenience: install the trap, and optionally a global timeout."""
    install_tree_kill_trap()
    install_global_timeout(timeout_seconds)


def install_from_env() -> None:
    """Install trap + opt-in timeout from ``ABEL_EDGE_TIMEOUT_SECONDS`` env.

    One-call convenience for CLI entry. Idempotent. ``ABEL_EDGE_TIMEOUT_SECONDS``
    value must be a positive integer to enable the wall-clock timeout.
    """
    install_tree_kill_trap()
    raw = os.environ.get("ABEL_EDGE_TIMEOUT_SECONDS", "").strip()
    if raw.isdigit():
        install_global_timeout(int(raw))
