"""Tests for ``causal_edge._runtime_harden``.

Verifies the three-layer joblib/fork deadlock prevention:
  - env vars set at apply() time
  - mp start method chosen
  - /proc descendant walk detects real children
  - signal trap installs without raising
  - global timeout is opt-in and idempotent

Signal-firing behavior is not exercised directly (would require a
subprocess rendezvous with controlled timing); we verify the installers
succeed and the supporting machinery is correct.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from causal_edge import _runtime_harden as rh


def test_apply_sets_thread_count_env_vars(monkeypatch):
    """apply() must set OMP/MKL/LOKY env vars to the documented defaults."""
    for key in rh._ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    # Reset the idempotency flag so apply() runs in this test.
    monkeypatch.setattr(rh, "_applied", False)

    rh.apply()

    for key, expected in rh._ENV_DEFAULTS.items():
        assert os.environ.get(key) == expected, f"{key} not set by apply()"


def test_apply_respects_user_overrides(monkeypatch):
    """setdefault semantics: if the user already set a value, apply() keeps it."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "16")
    monkeypatch.setattr(rh, "_applied", False)

    rh.apply()

    assert os.environ["LOKY_MAX_CPU_COUNT"] == "16"


def test_apply_is_idempotent(monkeypatch):
    """Multiple apply() calls must not raise (mp.set_start_method is once-only)."""
    monkeypatch.setattr(rh, "_applied", False)
    rh.apply()
    rh.apply()  # second call: _applied is True, early return
    rh.apply()


@pytest.mark.skipif(
    sys.platform == "win32", reason="mp start method set via forkserver is POSIX-only"
)
def test_apply_sets_forkserver_when_unset(monkeypatch):
    """apply() chooses forkserver on Linux/macOS — the actual cure for the hang.

    We cannot verify this against a real process because pytest shares the
    test runner's mp context. Instead, spawn a subprocess with a clean
    interpreter so the start method has not been set yet, and observe what
    apply() chooses.
    """
    script = (
        "import multiprocessing as mp;"
        "from causal_edge import _runtime_harden as rh;"
        "rh.apply();"
        "print(mp.get_start_method(allow_none=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(os.environ.get("PYTHONPATH", ""))},
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "forkserver"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc walk is Linux-only")
def test_descendant_pids_detects_real_child():
    """Fork a subprocess; verify _descendant_pids() finds it."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    try:
        # Small delay so the child is visible in /proc
        time.sleep(0.2)
        descendants = rh._descendant_pids(os.getpid())
        assert child.pid in descendants, f"expected child {child.pid} in descendants {descendants}"
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc walk is Linux-only")
def test_descendant_pids_empty_on_leaf():
    """A freshly spawned subprocess with no children reports empty descendants."""
    script = (
        "from causal_edge._runtime_harden import _descendant_pids;"
        "import os;"
        "print(len(_descendant_pids(os.getpid())))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert int(result.stdout.strip()) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="/proc walk is Linux-only")
def test_kill_descendants_reaps_real_child():
    """Fork a sleeper, call _kill_descendants, verify it dies promptly."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        time.sleep(0.2)
        count = rh._kill_descendants(grace_seconds=0.3)
        assert count >= 1, "expected at least one descendant killed"
        # Poll for child exit — should be fast after SIGKILL
        for _ in range(20):
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "child still alive after _kill_descendants"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_install_tree_kill_trap_is_idempotent(monkeypatch):
    monkeypatch.setattr(rh, "_trap_installed", False)
    rh.install_tree_kill_trap()
    rh.install_tree_kill_trap()  # second call: early return


def test_install_tree_kill_trap_registers_sigterm(monkeypatch):
    """After install, SIGTERM handler must be the module's custom one."""
    original = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(rh, "_trap_installed", False)
    try:
        rh.install_tree_kill_trap()
        current = signal.getsignal(signal.SIGTERM)
        assert current is not original, "SIGTERM handler was not replaced"
        assert callable(current)
    finally:
        signal.signal(signal.SIGTERM, original)


def test_install_global_timeout_zero_is_noop(monkeypatch):
    """timeout_seconds=0 must not install an alarm."""
    monkeypatch.setattr(rh, "_alarm_installed", False)
    rh.install_global_timeout(0)
    assert not rh._alarm_installed


def test_install_global_timeout_negative_is_noop(monkeypatch):
    monkeypatch.setattr(rh, "_alarm_installed", False)
    rh.install_global_timeout(-5)
    assert not rh._alarm_installed


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable (Windows)")
def test_install_global_timeout_installs_alarm(monkeypatch):
    """A positive timeout installs a SIGALRM handler."""
    original = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(rh, "_alarm_installed", False)
    try:
        rh.install_global_timeout(3600)  # long enough to never fire during test
        current = signal.getsignal(signal.SIGALRM)
        assert current is not original
        assert callable(current)
    finally:
        signal.alarm(0)  # cancel any pending alarm
        signal.signal(signal.SIGALRM, original)


def test_install_from_env_without_env_installs_only_trap(monkeypatch):
    monkeypatch.delenv("CAUSAL_EDGE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(rh, "_trap_installed", False)
    monkeypatch.setattr(rh, "_alarm_installed", False)

    rh.install_from_env()

    assert rh._trap_installed
    assert not rh._alarm_installed


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable (Windows)")
def test_install_from_env_installs_timeout_when_set(monkeypatch):
    monkeypatch.setenv("CAUSAL_EDGE_TIMEOUT_SECONDS", "3600")
    monkeypatch.setattr(rh, "_trap_installed", False)
    monkeypatch.setattr(rh, "_alarm_installed", False)
    original = signal.getsignal(signal.SIGALRM)
    try:
        rh.install_from_env()
        assert rh._alarm_installed
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)


def test_install_from_env_ignores_non_integer(monkeypatch):
    monkeypatch.setenv("CAUSAL_EDGE_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setattr(rh, "_trap_installed", False)
    monkeypatch.setattr(rh, "_alarm_installed", False)

    rh.install_from_env()

    assert rh._trap_installed
    assert not rh._alarm_installed
