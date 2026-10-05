"""Launch helper: wait until the Study H runner (given PID) has exited, then run Study G with the study venv.
Avoids running both studies' 7 workers at once on an 8-core machine. Logs to results_g/run_full.log."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def alive(pid: int) -> bool:
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259                                      # STILL_ACTIVE


def main() -> int:
    pid = int(sys.argv[1])
    while alive(pid):
        time.sleep(30)
    (HERE / "results_g").mkdir(exist_ok=True)
    with open(HERE / "results_g" / "run_full.log", "a", encoding="utf-8") as log:
        log.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Study H runner {pid} has exited; starting Study G\n")
        log.flush()
        return subprocess.call([str(HERE / ".venv" / "Scripts" / "python.exe"), "run_g.py", "--workers", "7"],
                               cwd=HERE, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    raise SystemExit(main())
