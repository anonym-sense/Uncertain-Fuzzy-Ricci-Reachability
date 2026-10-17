"""Run every Paper 1 job with a bounded process pool; one log per job in experiments/logs."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PY = str(HERE.parent / ".venv" / "Scripts" / "python.exe")
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)
MAXP = int(sys.argv[1]) if len(sys.argv) > 1 else 20

BUDGET = {"van_der_pol": 1024, "planar_quad": 1024}
jobs = []
for b in ("s_curve", "pendulum", "acc", "planar_quad", "van_der_pol"):
    for e in ("zono", "flow", "ricci", "box"):
        jobs.append((f"{b}__{e}", ["run_paper1.py", b, e, str(BUDGET.get(b, 512)), "curvature", "path"]))
for b in ("s_curve", "pendulum", "acc", "van_der_pol"):
    for sp in ("sensitivity", "width"):
        jobs.append((f"{b}__zono_{sp}", ["run_paper1.py", b, "zono", str(BUDGET.get(b, 512)), sp, "path"]))
for N in (1, 2, 3, 4, 6, 8, 10):
    for e in ("zono", "ricci"):
        jobs.append((f"chain{N}__{e}", ["run_paper1.py", f"chain{N}", e, "256", "curvature", "path"]))
for N in (8, 10):
    jobs.append((f"chain{N}__mc", ["run_paper1.py", f"chain{N}", "mc"]))
jobs.append(("landscape", ["landscape.py"]))
jobs.append(("replication", ["replication_2020.py", "0.3"]))

# longest first
order = {"van_der_pol": 0, "planar_quad": 1, "landscape": 2}
jobs.sort(key=lambda j: min([v for k, v in order.items() if k in j[0]] + [9]))

running = []
t0 = time.time()
while jobs or running:
    while jobs and len(running) < MAXP:
        name, args = jobs.pop(0)
        f = open(LOGS / f"{name}.log", "w")
        running.append((name, subprocess.Popen([PY, *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT), f))
    for r in running[:]:
        if r[1].poll() is not None:
            r[2].close()
            running.remove(r)
            print(f"[{time.time() - t0:7.0f}s] finished {r[0]} (rc={r[1].returncode}); "
                  f"{len(running)} running, {len(jobs)} queued", flush=True)
    time.sleep(2)
print("ALL DONE", flush=True)
