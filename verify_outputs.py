#!/usr/bin/env python3
import json
import sys
from pathlib import Path

EXPECTED = {"5638": (1063, 67), "6005": (153, 10), "5645": (411, 26)}

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
actual = {str(row["task_id"]): (row["num_timesteps"], row["num_chunks"]) for row in rows}
if actual != EXPECTED:
    raise SystemExit(f"verification failed: expected={EXPECTED}, actual={actual}")
print(f"PASS: {actual}")

