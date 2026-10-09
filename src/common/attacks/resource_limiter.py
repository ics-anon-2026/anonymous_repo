# -*- coding: utf-8 -*-
"""资源上限封装：把实验进程的 CPU/内存占用压到用户设定的上限（默认 70%）。

用法：
    # 在脚本最顶部 import 并调用
    from resource_limiter import apply_limits
    apply_limits(cpu_frac=0.70, mem_frac=0.70)

作用：
- CPU：torch/OMP/MKL/OpenBLAS 线程数 = floor(逻辑核 × cpu_frac)，并用 psutil 把进程
       亲和性 pin 到前 N 个核（其余核留给用户/系统）。
- 内存：Windows 下创建 Job Object 并设 JobMemoryLimit = 物理内存 × mem_frac（硬上限，
       超限进程被杀，防止 OOM 拖垮整机）；非 Windows 下退化为软预测拒绝（见 guard）。
- 提供 predicted_memory_guard(max_bytes)：在重型任务前调用，预测超限则抛错阻断。

注意：线程亲和性对稠密矩阵乘（Metattack/训练）的限制最有效；内存上限是兜底保险，
      因为 PRBCD/victim 训练本身峰值仅 2–6GB，远低于 70% 上限。
"""
import os
import sys
import platform


def apply_limits(cpu_frac: float = 0.70, mem_frac: float = 0.70, verbose: bool = True):
    """应用 CPU 与内存限制。返回实际生效的 (n_threads, mem_cap_bytes)。"""
    n_threads = _limit_cpu(cpu_frac, verbose)
    mem_cap = _limit_memory(mem_frac, verbose)
    return n_threads, mem_cap


def _limit_cpu(cpu_frac: float, verbose: bool) -> int:
    import torch  # 延迟 import，避免无 torch 环境报错
    n_logical = os.cpu_count() or 1
    n_threads = max(1, int(n_logical * cpu_frac))
    # 1) 限制各 BLAS / OpenMP 后端线程数（覆盖 numpy/torch/scipy）
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)
    # torch 全局线程
    try:
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    # 2) 进程 CPU 亲和性：pin 到前 n_threads 个逻辑核
    try:
        import psutil
        p = psutil.Process()
        all_cores = list(range(n_logical))
        pinned = all_cores[:n_threads]
        p.cpu_affinity(pinned)
        if verbose:
            print(f"[resource_limiter] CPU: 限 {n_threads}/{n_logical} 线程 (={cpu_frac:.0%}); "
                  f"亲和性 pin 到核 {pinned}")
    except Exception as e:
        if verbose:
            print(f"[resource_limiter] CPU 亲和性设置失败（已设线程数）: {type(e).__name__}: {e}")
    return n_threads


def _limit_memory(mem_frac: float, verbose: bool):
    try:
        import ctypes
        total = _total_phys_bytes()
        cap = int(total * mem_frac)
        if platform.system() == "Windows":
            _win_job_memory_limit(cap, verbose)
        else:
            # 非 Windows：用 resource.setrlimit(RLIMIT_AS) 做软上限（Linux/macOS）
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))  # (soft, hard)
                if verbose:
                    print(f"[resource_limiter] 内存软上限 RLIMIT_AS = {cap/1e9:.1f} GB ({mem_frac:.0%})")
            except Exception as e:
                if verbose:
                    print(f"[resource_limiter] 非 Windows 内存硬上限不可用（仅软预测）: {e}")
        return cap
    except Exception as e:
        if verbose:
            print(f"[resource_limiter] 内存限制设置失败: {e}")
        return 0


def _total_phys_bytes() -> int:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullTotalPhys


def _win_job_memory_limit(cap_bytes: int, verbose: bool):
    """Windows Job Object：把当前进程放入一个带 JobMemoryLimit 的 Job，硬上限内存。"""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32

    # 创建 Job
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        if verbose:
            print("[resource_limiter] CreateJobObject 失败")
        return

    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    # typedef struct {
    #   BASIC_LIMIT_INFORMATION BasicLimitInformation;
    #   IO_COUNTERS IoInfo;
    #   SIZE_T ProcessMemoryLimit;     // 与 JobMemoryLimit 二选一，设为 0 用下者
    #   SIZE_T JobMemoryLimit;
    #   SIZE_T PeakProcessMemoryUsed;
    #   SIZE_T PeakJobMemoryUsed;
    # } JOBOBJECT_EXTENDED_LIMIT_INFORMATION;
    # BasicLimitInformation 含 enumeration kernel time(8) + user time(8) + ... + LimitFlags(4) + 等
    # 我们用 ctypes 直接拼布局（仅在 x64 Windows 验证）。
    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(f"v{i}", ctypes.c_ulonglong) for i in range(6)]

    class JOB_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_void_p),
            ("MaximumWorkingSetSize", ctypes.c_void_p),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_void_p),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOB_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_void_p),
            ("JobMemoryLimit", ctypes.c_void_p),
            ("PeakProcessMemoryUsed", ctypes.c_void_p),
            ("PeakJobMemoryUsed", ctypes.c_void_p),
        ]

    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY
    info.JobMemoryLimit = ctypes.c_void_p(cap_bytes)

    ok = kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(info), ctypes.sizeof(info))  # 9 = JobObjectExtendedLimitInformation
    if not ok:
        if verbose:
            print("[resource_limiter] SetInformationJobObject 失败 (GetLastError=%s)" % kernel32.GetLastError())
        return

    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        if verbose:
            print("[resource_limiter] AssignProcessToJobObject 失败 (GetLastError=%s)" % kernel32.GetLastError())
        return
    if verbose:
        print(f"[resource_limiter] 内存 Job Object 硬上限 = {cap_bytes/1e9:.1f} GB ({cap_bytes/(_total_phys_bytes() or 1):.0%})")


def predicted_memory_guard(max_bytes: int, what: str = "本任务"):
    """在重型任务前调用：若预测内存 > max_bytes 则抛 RuntimeError 阻断（防止整机和 OOM）。"""
    import ctypes
    import psutil
    avail = psutil.virtual_memory().available
    if max_bytes > avail:
        raise RuntimeError(
            f"[resource_limiter] 预测 {what} 需 ~{max_bytes/1e9:.1f} GB，但当前可用仅 {avail/1e9:.1f} GB，"
            f"拒绝执行以免拖垮机器。建议缩小规模或换服务器。")
    if max_bytes > int(_total_phys_bytes() * 0.70):
        raise RuntimeError(
            f"[resource_limiter] 预测 {what} 需 ~{max_bytes/1e9:.1f} GB，超过 70% 上限，拒绝执行。")
    return True


if __name__ == "__main__":
    n, cap = apply_limits(0.70, 0.70)
    import torch
    x = torch.randn(1000, 1000)
    torch.set_num_threads(n)
    print("test matmul threads:", torch.get_num_threads())
