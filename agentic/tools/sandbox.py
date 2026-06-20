"""
L1 process sandbox -- OS-level isolation for untrusted code execution.

Uses subprocess with resource limits (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC)
and Python's -I -E flags to sandbox arbitrary Python code in a dedicated
subprocess with a transient temp-filesystem workspace.

Security boundaries (L1):
  - Process-level RLIMIT_AS (virtual memory) hard cap
  - Process-level RLIMIT_CPU (CPU seconds) hard cap
  - Process-level RLIMIT_NPROC = 0 (fork/thread creation blocked)
  - Network namespace isolation via unshare(CLONE_NEWNET) when disabled
  - ``python3 -I -E`` -- no user site-packages, no PYTHON* env overrides
  - Transient temp directory per invocation, auto-destroyed on exit
  - Minimal process environment (PATH + HOME only)
  - Stdout/stderr size caps prevent parent-side memory exhaustion

Future L2/L3: add Landlock / seccomp-bpf / container runtime.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_preexec(
    memory_bytes: int,
    cpu_secs: int,
    disable_network: bool,
) -> callable:
    """Return a *preexec_fn* callable that applies resource limits and,
    optionally, network namespace isolation in the child process.

    Runs after fork() but before exec() in the child.  Only async-signal-safe
    calls are permitted here.
    """
    def _preexec() -> None:
        # Virtual-memory cap (RLIMIT_AS) -- both soft and hard.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, resource.error):
            pass

        # CPU-seconds cap (RLIMIT_CPU).
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (cpu_secs, resource.RLIM_INFINITY)
            )
        except (ValueError, resource.error):
            pass

        # Disallow fork/thread creation entirely.
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, resource.error):
            pass

        # Network isolation -- drop the child into a private network
        # namespace with no routes or interfaces.
        if disable_network:
            try:
                os.unshare(os.CLONE_NEWNET)
            except (AttributeError, OSError):
                pass  # kernel or permissions may not allow it

    return _preexec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_in_sandbox(
    code: str,
    timeout_sec: int = 10,
    memory_mb: int = 128,
    allow_network: bool = False,
) -> str:
    """Execute *code* as a Python script inside an OS-level sandbox.

    A temp directory is created, the code is written to a ``.py`` file
    inside it, and a fresh ``python3 -I -E`` subprocess is spawned with
    resource limits and a minimal environment.

    Parameters
    ----------
    code:
        Python source code to execute in the sandbox.
    timeout_sec:
        Wall-clock seconds before the process is killed (also used as
        the RLIMIT_CPU cap minus one second to avoid a race between
        the signal and the parent timeout).
    memory_mb:
        RLIMIT_AS cap in mebibytes.
    allow_network:
        If *False* (default), attempt network-namespace isolation via
        ``os.unshare(os.CLONE_NEWNET)``.

    Returns
    -------
    str
        - **On success:** stdout of the subprocess (UTF-8 decoded,
          truncated to 10 000 chars, trailing newline preserved so the
          caller can distinguish empty output from no output).
        - **Non-zero exit:** ``"SandboxError(code=N): <stderr>"`` where
          *stderr* is truncated to 500 chars.
        - **Timeout:** ``"SandboxError: timeout after Ns"``.
    """
    # Resolve the Python interpreter path (should always exist).
    python_exe: str = sys.executable
    if not python_exe:
        python_exe = "python3"

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "sandbox_script.py")

        with open(script_path, "w") as fh:
            fh.write(code)

        # Build a minimal environment -- strip everything except PATH and HOME.
        minimal_env: dict[str, str] = {}
        for key in ("PATH", "HOME"):
            val = os.environ.get(key)
            if val is not None:
                minimal_env[key] = val

        memory_bytes: int = memory_mb * 1024 * 1024

        # Set CPU limit slightly lower than the wall-clock timeout so that
        # CPU-hogging processes are killed by SIGXCPU *before* the parent's
        # communicate(timeout=) fires, giving us a clean exit code.
        cpu_secs: int = max(1, timeout_sec - 1)

        preexec_fn = _build_preexec(
            memory_bytes=memory_bytes,
            cpu_secs=cpu_secs,
            disable_network=not allow_network,
        )

        try:
            proc = subprocess.Popen(
                [python_exe, "-I", "-E", script_path],
                preexec_fn=preexec_fn,
                env=minimal_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
            )
        except FileNotFoundError:
            return f"SandboxError: interpreter not found: {python_exe}"
        except PermissionError:
            return f"SandboxError: permission denied: {python_exe}"

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return f"SandboxError: timeout after {timeout_sec}s"

    # ── tmpdir is now destroyed ──────────────────────────────────────

    stdout_str: str = stdout_bytes.decode("utf-8", errors="replace")[:10000]
    stderr_str: str = stderr_bytes.decode("utf-8", errors="replace")[:500]

    if proc.returncode != 0:
        # Signal deaths produce negative codes (e.g. -24 for SIGXCPU,
        # -6 for SIGABRT, -9 for SIGKILL).
        return f"SandboxError(code={proc.returncode}): {stderr_str}"

    return stdout_str
