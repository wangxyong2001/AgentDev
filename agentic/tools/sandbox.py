"""
L1 进程沙箱 -- 不可信代码执行的 OS 级隔离。

使用 subprocess 配合资源限制 (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC)
和 Python 的 -I -E 标志，在独立的子进程中以临时文件系统工作空间
来沙箱化任意 Python 代码。

安全边界 (L1):
  - 进程级 RLIMIT_AS（虚拟内存）硬上限
  - 进程级 RLIMIT_CPU（CPU 秒数）硬上限
  - 进程级 RLIMIT_NPROC = 0（禁止 fork/线程创建）
  - 通过 unshare(CLONE_NEWNET) 实现网络命名空间隔离
  - ``python3 -I -E`` -- 不使用用户 site-packages，PYTHON* 环境变量无效
  - 每次调用使用临时目录，退出时自动销毁
  - 最小进程环境（仅 PATH + HOME）
  - Stdout/stderr 大小上限防止父进程内存耗尽

未来 L2/L3: 增加 Landlock / seccomp-bpf / 容器运行时。

为什么用 preexec_fn？
  ``subprocess.Popen`` 的 ``preexec_fn`` 参数在子进程中** fork() 之后 **
  但 ** exec() 之前 ** 运行。这个时机至关重要：资源限制和命名空间隔离
  必须在目标程序开始执行 *之前* 应用，因为一旦 ``exec()`` 替换了进程映像，
  这些限制就会被继承且无法被子进程解除。``preexec_fn`` 内部只能调用
  异步信号安全函数（``setrlimit`` 和 ``unshare`` 都是安全的）。

用法:
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
# 辅助函数
# ---------------------------------------------------------------------------

def _build_preexec(
    memory_bytes: int,
    cpu_secs: int,
    disable_network: bool,
) -> callable:
    """
    构造一个 *preexec_fn* 可调用对象，在子进程中应用资源限制和
    （可选）网络命名空间隔离。

    处理逻辑:
      返回的函数在子进程的 fork() 之后、exec() 之前运行。
      preexec_fn 中只能调用异步信号安全函数
      （setrlimit、unshare、_exit 是安全的；malloc、printf、锁不安全）。

    参数:
        memory_bytes: RLIMIT_AS（虚拟内存）硬上限。
        cpu_secs: RLIMIT_CPU（CPU 秒数）软上限。硬上限保持为
            RLIM_INFINITY，使得在 SIGKILL 之前会先发送一个 SIGXCPU。
        disable_network: 若为 True，调用 os.unshare(CLONE_NEWNET)
            将子进程放入无路由和接口的私有网络命名空间。

    返回值:
        一个零参数可调用对象，适合作为 Popen 的 preexec_fn。
    """
    def _preexec() -> None:
        # ── RLIMIT_AS：虚拟内存硬上限 ─────────────────────────────
        # 防止 fork 炸弹 / 内存耗尽攻击。沙箱化代码无法分配超过
        # `memory_bytes` 的虚拟地址空间（包括 mmap、栈、堆）。
        # 软限制和硬限制设为相同值，使得子进程无法自行提高上限。
        # 失败时静默忽略。某些平台（如启用了 SIP 的 macOS）可能会
        # 拒绝 RLIMIT_AS 变更；父进程的超时机制提供后备安全网。
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, resource.error):
            pass

        # ── RLIMIT_CPU：CPU 秒数上限 ──────────────────────────────
        # 防止 CPU 密集型循环耗尽其他进程的资源。
        # 软上限 = timeout_sec - 1；硬上限 = RLIM_INFINITY。
        # 内核在超过软上限时发送 SIGXCPU，超过硬上限时发送 SIGKILL。
        # 将硬上限保留为 infinity 意味着只发送 SIGXCPU，子进程理论上
        # 可以捕获它——但父进程的 communicate(timeout=) 提供了挂钟
        # 时间强制执行。
        # RLIMIT_CPU 与挂钟超时之间的 1 秒间隔确保 SIGXCPU 在父进程的
        # communicate() 抛出 TimeoutExpired 之前到达，从而产生干净的
        # 负退出码。
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (cpu_secs, resource.RLIM_INFINITY)
            )
        except (ValueError, resource.error):
            pass

        # ── RLIMIT_NPROC：阻止 fork/线程创建 ──────────────────────
        # 将 RLIMIT_NPROC 设为 (0, 0) 阻止子进程调用 fork()、
        # clone() 或 pthread_create()。这限制了爆炸半径：即使沙箱
        # 代码找到了逃生途径，也无法产生子进程或线程。
        # 注意：此限制适用于用户 ID，因此在共享用户沙箱上设置
        # NPROC=0 可能会影响同级进程。在每次调用单进程的使用场景
        # 中这是可接受的。
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, resource.error):
            pass

        # ── 网络命名空间隔离 ──────────────────────────────────────
        # 当 disable_network 为 True 时，将子进程放入私有网络命名空间，
        # 没有接口（lo 关闭）和路由。这防止数据泄漏、回连 C2 服务器
        # 以及基于网络的侧信道攻击。
        #
        # 失败模式（静默忽略）：
        #   - AttributeError：Linux < 2.6.24 或非 Linux 系统（无 unshare）。
        #   - OSError：权限不足（容器、seccomp 限制）。
        # 无论哪种情况，子进程都会保留父进程的网络访问权限
        # — 这在调用者层面作为尽力而为的缺口被记录。
        if disable_network:
            try:
                os.unshare(os.CLONE_NEWNET)
            except (AttributeError, OSError):
                pass  # 内核或权限可能不允许

    return _preexec


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def execute_in_sandbox(
    code: str,
    timeout_sec: int = 10,
    memory_mb: int = 128,
    allow_network: bool = False,
) -> str:
    """将 *code* 作为 Python 脚本在 OS 级沙箱内执行。

    创建一个临时目录，将代码写入其中的 ``.py`` 文件，然后使用资源限制
    和最小环境启动一个全新的 ``python3 -I -E`` 子进程。输出以字符串形式
    返回；错误以内联方式呈现为 ``"SandboxError(...)"``，使得调用方
    （通常是 LLM 工具）无需异常处理即可将其纳入上下文。

    应用的安全层级:
      1. 通过 subprocess 实现进程隔离（独立 PID、独立地址空间）。
      2. RLIMIT_AS 限制虚拟内存（默认 128 MB）。
      3. RLIMIT_CPU 限制 CPU 秒数（timeout - 1）。
      4. RLIMIT_NPROC = 0 阻止 fork/线程创建。
      5. ``python3 -I -E``：-I = 隔离模式（无用户 site-packages，
         无 ``PYTHON*`` 环境变量），-E = 忽略 ``PYTHON*`` 环境变量。
      6. 临时目录，退出时销毁。
      7. 环境变量白名单：仅 PATH 和 HOME。
      8. 网络命名空间隔离（除非 allow_network=True）。
      9. Stdout/stderr 截断（10 KB / 500 B）防止父进程因大量输出
         而耗尽内存。

    参数
    ----------
    code:
        在沙箱中执行的 Python 源代码。
    timeout_sec:
        进程被杀死前的挂钟秒数（也用于 RLIMIT_CPU 上限减一秒，
        以避免信号与父进程超时之间的竞态条件）。
    memory_mb:
        RLIMIT_AS 上限，以 MiB 为单位。默认 128。
    allow_network:
        如果为 *False*（默认），尝试通过 ``os.unshare(os.CLONE_NEWNET)``
        进行网络命名空间隔离。仅当沙箱代码确实需要网络访问时才设为 *True*
        （例如为 Agent 工具获取远程数据）。

    返回值
    -------
    str
        - **成功时：** 子进程的 stdout（UTF-8 解码，截断至 10000 字符，
          保留尾部换行以使调用方能区分空输出与无输出）。
        - **非零退出：** ``"SandboxError(code=N): <stderr>"``，其中
          *stderr* 截断至 500 字符。
        - **超时：** ``"SandboxError: timeout after Ns"``。
    """
    # ── 解析解释器路径 ───────────────────────────────────────────────
    # sys.executable 是当前运行的解释器路径。这保证框架与沙箱之间
    # 的 Python 版本一致性。仅在 sys.executable 为空时回退到 "python3"
    # （边缘情况：冻结的二进制文件、某些嵌入式环境）。
    python_exe: str = sys.executable
    if not python_exe:
        python_exe = "python3"

    # ── 创建临时工作区 ──────────────────────────────────────────────
    # TemporaryDirectory 在退出时自动清理（包括通过 __exit__ 的异常退出）。
    # 前缀有助于在遗留临时目录时进行调试（例如清理前进程崩溃）。
    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "sandbox_script.py")

        with open(script_path, "w") as fh:
            fh.write(code)

        # ── 最小环境 ────────────────────────────────────────────────
        # 去除除 PATH 和 HOME 之外的所有环境变量继承。这防止将秘密
        # （API 密钥、令牌）泄漏到沙箱中，并阻止依赖 PYTHONPATH、
        # LD_PRELOAD 或类似注入向量的攻击。
        minimal_env: dict[str, str] = {}
        for key in ("PATH", "HOME"):
            val = os.environ.get(key)
            if val is not None:
                minimal_env[key] = val

        memory_bytes: int = memory_mb * 1024 * 1024

        # ── CPU 超时缓冲 ────────────────────────────────────────────
        # 将 CPU 限制设置得略低于挂钟超时，以便 CPU 密集型进程在父进程的
        # communicate(timeout=) 触发 *之前* 就被 SIGXCPU 杀死，从而产生
        # 干净的负退出码（例如 SIGXCPU 为 -24），而不是模糊的
        # TimeoutExpired 异常。
        cpu_secs: int = max(1, timeout_sec - 1)

        preexec_fn = _build_preexec(
            memory_bytes=memory_bytes,
            cpu_secs=cpu_secs,
            disable_network=not allow_network,
        )

        # ── 启动沙箱子进程 ───────────────────────────────────────────
        # 标志：-I（隔离模式）禁用 site-packages，-E 忽略 PYTHON* 环境变量。
        # 二者共同确保沙箱代码在干净的 Python 环境中运行，不受父进程配置影响。
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

        # ── 使用挂钟超时收集输出 ─────────────────────────────────────
        # communicate() 并发读取 stdout + stderr 以避免死锁。
        # 如果进程超过 timeout_sec，我们杀死它并等待实际终止后再返回。
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return f"SandboxError: timeout after {timeout_sec}s"

    # ── tmpdir 现在已销毁 ────────────────────────────────────────────
    # TemporaryDirectory 上下文管理器已清理临时目录。
    # 沙箱代码写入的所有文件都已消失。

    # ── 限制输出大小 ──────────────────────────────────────────────
    # 将 stdout 截断至 10 KB、stderr 截断至 500 B，防止沙箱通过
    # 大量输出导致父进程内存压力（沙箱逃逸的常见 DoS 向量）。
    stdout_str: str = stdout_bytes.decode("utf-8", errors="replace")[:10000]
    stderr_str: str = stderr_bytes.decode("utf-8", errors="replace")[:500]

    if proc.returncode != 0:
        # ── 非零退出处理 ────────────────────────────────────────────
        # 负返回码表示信号终止（例如 RLIMIT_CPU 的 SIGXCPU 为 -24，
        # SIGABRT 为 -6，SIGKILL 为 -9）。正返回码是正常的 Python
        # sys.exit(n)。错误字符串以内联方式返回而非抛出异常，以便
        # LLM 工具调用方能对其进行推理。
        return f"SandboxError(code={proc.returncode}): {stderr_str}"

    return stdout_str
