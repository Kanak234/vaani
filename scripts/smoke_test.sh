#!/usr/bin/env bash
set -euo pipefail
echo '=== Vaani Smoke Test ==='

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HERE/.venv"
PYTHON="$VENV/bin/python"

# 1. Hardware detection
echo -n 'Hardware detection... '
$PYTHON -c '
from vaani.system.hardware import full_report
r = full_report()
print(f"OK — {r.os_name}, {r.cpu.model}, {r.ram_total_mb}MB RAM")
if r.has_nvidia:
    print(f"  GPU: {r.primary_gpu.name} ({r.primary_gpu.vram_total_mb}MB)")
else:
    print("  No NVIDIA GPU detected")
'

# 2. Profile
echo -n 'Capability profile... '
$PYTHON -c '
from vaani.system.hardware import full_report
from vaani.system.profile import determine_profile
r = full_report()
p = determine_profile(r)
print(f"OK — {p.level.name}")
for d in p.decisions:
    print(f"  {d.component}: {d.device} ({d.reason})")
'

# 3. Config
echo -n 'Config system... '
$PYTHON -c 'from vaani.config import VaaniConfig; c = VaaniConfig(); print(f"OK — model_dir={c.model_dir}")'

# 4. Startup discovery
echo 'Startup discovery...'
$PYTHON -c '
from vaani.diagnostics.startup import run_startup_discovery
def progress(check):
    print(f"  {check.icon} {check.name}: {check.detail}")
r = run_startup_discovery(on_progress=progress)
print(f"Ready: {r.ready} ({r.total_ms:.0f}ms)")
'

echo '=== Smoke Test Complete ==='
