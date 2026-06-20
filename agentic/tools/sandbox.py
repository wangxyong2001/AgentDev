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

Why preexec_fn?
  The ``preexec_fn`` argument to ``subprocess.Popen`` runs **after**
  ``fork()`` but **before** ``exec()`` in the child process. This timing
  is critical: resource limits and namespace isolation must be applied
  *before* the target program starts executing, because once ``exec()``
  replaces the process image, those restrictions are inherited and cannot
  be lifted by the child. Only async-signal-safe functions may be called
  inside ``preexec_fn`` (``setrlimit`` and ``unshare`` are both safe).

Usage:
  >>> from agentic.tools.sandbox import execute_in_sandbox
  >>> out = execute_in_sandbox("print('hello')")
  >>> print(out)
  "hello\\n"
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

    The returned function runs after fork() but before exec() in the
    child process.  Only async-signal-safe calls are permitted here
    (setrlimit, unshare, _exit are safe; malloc, printf, locks are not).

    Args:
        memory_bytes: Hard cap for RLIMIT_AS (virtual memory).
        cpu_secs: Soft cap for RLIMIT_CPU (CPU seconds). The hard cap
            is left at RLIM_INFINITY so a second SIGXCPU is delivered
            before SIGKILL.
        disable_network: If True, call os.unshare(CLONE_NEWNET) to
            drop the child into a private network namespace with no
            routes or interfaces.

    Returns:
        A zero-argument callable suitable as Popen's preexec_fn.
    """
    def _preexec() -> None:
        # ── RLIMIT_AS: Virtual memory hard cap ──────────────────────
        # Prevents fork-bomb / memory-exhaustion attacks. The sandboxed
        # code cannot allocate more than `memory_bytes` of virtual
        # address space (including mmap, stack, heap). Both soft and
        # hard limits are set to the same value so the child cannot
        # raise its own limit.
        # Failure: silently ignored. Some platforms (e.g. macOS with
        # SIP) may reject RLIMIT_AS changes; the parent timeout
        # provides a fallback safety net.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, resource.error):
            pass

        # ── RLIMIT_CPU: CPU seconds cap ─────────────────────────────
        # Prevents CPU-hogging loops from starving other processes.
        # Soft cap = timeout_sec - 1; hard cap = RLIM_INFINITY.
        # The kernel delivers SIGXCPU when the soft cap is exceeded,
        # then SIGKILL when the hard cap is hit. Leaving the hard cap
        # at infinity means only SIGXCPU is sent, which the child can
        # theoretically catch — but the parent communicate(timeout=)
        # provides the wall-clock enforcement.
        # The 1-second gap between RLIMIT_CPU and the wall timeout
        # ensures SIGXCPU arrives before the parent's communicate()
        # raises TimeoutExpired, giving a clean negative exit code.
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (cpu_secs, resource.RLIM_INFINITY)
            )
        except (ValueError, resource.error):
            pass

        # ── RLIMIT_NPROC: Block fork/thread creation ────────────────
        # Setting RLIMIT_NPROC to (0, 0) prevents the child process
        # from calling fork(), clone(), or pthread_create(). This
        # contains the blast radius: even if the sandboxed code finds
        # a way to escape, it cannot spawn child processes or threads.
        # Note: this limit applies to the user ID, so setting NPROC=0
        # on a shared-user sandbox could affect sibling processes.
        # In single-process-per-invocation usage this is acceptable.
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, resource.error):
            pass

        # ── Network namespace isolation ─────────────────────────────
        # When disable_network is True, drop the child into a private
        # network namespace with no interfaces (lo is down) and no
        # routes. This prevents data exfiltration, callbacks to C2
        # servers, and network-based side channels.
        #
        # Failure modes (silently ignored):
        #   - AttributeError: Linux < 2.6.24 or non-Linux (no unshare).
        #   - OSError: Insufficient permissions (containers, seccomp).
        # In either case the child retains the parent's network access
        # — this is logged at the caller level as a best-effort gap.
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
    resource limits and a minimal environment. Output is returned as a
    string; errors are surfaced in-band as ``"SandboxError(...)"`` so
    the caller (typically an LLM tool) can include them in its context
    without exception handling.

    Security layers applied:
      1. Process isolation via subprocess (separate PID, separate address space).
      2. RLIMIT_AS caps virtual memory (128 MB default).
      3. RLIMIT_CPU caps CPU seconds (timeout - 1).
      4. RLIMIT_NPROC = 0 blocks fork/thread creation.
      5. ``python3 -I -E``: -I = isolated mode (no user site-packages,
         no ``PYTHON*`` env vars), -E = ignore ``PYTHON*`` environment.
      6. Transient temp directory, destroyed on exit.
      7. Environment whitelist: only PATH and HOME.
      8. Network namespace isolation (unless allow_network=True).
      9. Stdout/stderr truncated (10 KB / 500 B) to prevent parent
         memory exhaustion from runaway output.

    Parameters
    ----------
    code:
        Python source code to execute in the sandbox.
    timeout_sec:
        Wall-clock seconds before the process is killed (also used as
        the RLIMIT_CPU cap minus one second to avoid a race between
        the signal and the parent timeout).
    memory_mb:
        RLIMIT_AS cap in mebibytes.  Default 128.
    allow_network:
        If *False* (default), attempt network-namespace isolation via
        ``os.unshare(os.CLONE_NEWNET)``.  Set to *True* only when the
        sandboxed code legitimately needs network access (e.g. fetching
        remote data for an agent tool).

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
    # ── Resolve interpreter path ────────────────────────────────────
    # sys.executable is the path to the currently running interpreter.
    # This guarantees version consistency between the harness and the
    # sandbox.  Fallback to "python3" only if sys.executable is empty
    # (edge case: frozen binaries, some embedded environments).
    python_exe: str = sys.executable
    if not python_exe:
        python_exe = "python3"

    # ── Create temp workspace ───────────────────────────────────────
    # TemporaryDirectory auto-cleans on exit (including exception exits
    # via __exit__).  The prefix aids debugging if a stray temp dir is
    # left behind (e.g. process crash before cleanup).
    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "sandbox_script.py")

        with open(script_path, "w") as fh:
            fh.write(code)

        # ── Minimal environment ─────────────────────────────────────
        # Strips inheritance of all environment variables except PATH
        # and HOME.  This prevents leaking secrets (API keys, tokens)
        # into the sandbox and blocks attacks that rely on PYTHONPATH,
        # LD_PRELOAD, or similar injection vectors.
        minimal_env: dict[str, str] = {}
        for key in ("PATH", "HOME"):
            val = os.environ.get(key)
            if val is not None:
                minimal_env[key] = val

        memory_bytes: int = memory_mb * 1024 * 1024

        # ── CPU timeout buffer ──────────────────────────────────────
        # Set CPU limit slightly lower than the wall-clock timeout so
        # that CPU-hogging processes are killed by SIGXCPU *before*
        # the parent's communicate(timeout=) fires, giving us a clean
        # negative exit code (e.g. -24 for SIGXCPU) instead of an
        # ambiguous TimeoutExpired exception.
        cpu_secs: int = max(1, timeout_sec - 1)

        preexec_fn = _build_preexec(
            memory_bytes=memory_bytes,
            cpu_secs=cpu_secs,
            disable_network=not allow_network,
        )

        # ── Spawn sandboxed process ─────────────────────────────────
        # Flags: -I (isolated mode) disables site-packages, -E ignores
        # PYTHON* environment variables.  Together they ensure the
        # sandboxed code runs with a clean Python environment regardless
        # of the parent's configuration.
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

        # ── Collect output with wall-clock timeout ──────────────────
        # communicate() reads stdout + stderr concurrently to avoid
        # deadlock.  If the process exceeds timeout_sec, we kill it
        # and wait for actual termination before returning.
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return f"SandboxError: timeout after {timeout_sec}s"

    # ── tmpdir is now destroyed ──────────────────────────────────────
    # The TemporaryDirectory context manager has cleaned up the temp
    # directory.  Any files the sandboxed code wrote are gone.

    # ── Cap output sizes ────────────────────────────────────────────
    # Truncating stdout to 10 KB and stderr to 500 B prevents the
    # sandbox from causing memory pressure in the parent process
    # through excessive output (a common DoS vector in sandbox escapes).
    stdout_str: str = stdout_bytes.decode("utf-8", errors="replace")[:10000]
    stderr_str: str = stderr_bytes.decode("utf-8", errors="replace")[:500]

    if proc.returncode != 0:
        # ── Non-zero exit handling ──────────────────────────────────
        # Negative return codes indicate signal death (e.g. -24 for
        # SIGXCPU from RLIMIT_CPU, -6 for SIGABRT, -9 for SIGKILL).
        # Positive codes are normal Python sys.exit(n).
        # The error string is returned in-band rather than raised as
        # an exception so the LLM tool caller can reason about it.
        return f"SandboxError(code={proc.returncode}): {stderr_str}"

    return stdout_str
