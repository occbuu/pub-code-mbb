"""Wait for this batch process, then analyze once. Not a recurring job."""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pid = int(sys.argv[1])
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.restype = ctypes.c_void_p
kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
kernel.CloseHandle.argtypes = [ctypes.c_void_p]
handle = kernel.OpenProcess(0x1000, False, pid)
status = root / "logs/13b_local_upgrade.status"
if not handle:
    raise OSError(ctypes.get_last_error(), "Cannot monitor batch process")
status.write_text(f"RUNNING: inference PID {pid}; final analysis will run automatically.\n")
try:
    while True:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), "Cannot query batch process")
        if code.value != 259: break
        time.sleep(10)
finally:
    kernel.CloseHandle(handle)
lines = (root / "logs/13b_local_upgrade.log").read_text(encoding="utf-8").splitlines()
if not lines or lines[-1] != "COMPLETE":
    status.write_text("INCOMPLETE: inspect inference logs and resume the identical command.\n")
    sys.exit(1)
with (root / "logs/13b_local_analysis.log").open("w", encoding="utf-8") as output:
    result = subprocess.run([sys.executable, str(root / "scripts/analyze_13b_upgrade.py")], stdout=output, stderr=subprocess.STDOUT)
status.write_text("COMPLETE: inference and analysis finished.\n" if result.returncode == 0 else "INFERENCE COMPLETE; ANALYSIS FAILED: inspect analysis log.\n")
sys.exit(result.returncode)
