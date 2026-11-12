#!/usr/bin/env python3
"""Generate the corrupt-sensor-recon fixtures: sensor_alpha.log + sensor_beta.csv.

Deterministic by design: the RNG is seeded so re-running produces byte-identical
files. Beta uses seeded jitter; Alpha is a hand-designed sequence.

The data spans exactly 15 one-minute UTC buckets, 14:30 through 14:44 on
2024-05-12, and embeds the edge cases the task is built around:

  * THERMAL-INERTIA GLITCH FILTER (Alpha): the bath cannot move > 1.5 C/min, so
    any reading whose rate of change from the LAST ACCEPTED reading exceeds that
    is a glitch -> temp dropped (timestamp kept). The glitch cluster:
      14:30 20.0 (first -> accepted) ; 14:31 20.5 (0.5/min -> valid)
      14:32 35.0 (14.5/min -> GLITCH) ; 14:33 36.0 (7.75/min vs 14:31 -> GLITCH)
      14:34 22.0 (1.5 C over 3 min vs 14:31 = 0.5/min -> valid)
    The trap: 14:34 must be compared to the last VALID reading (14:31, 20.5), not
    the last reading (14:33, 36.0) — the latter wrongly flags 14:34 as a glitch.
    The two glitch minutes (14:32/14:33) then interpolate from 14:31 and 14:34.
  * Non-200 Alpha row at the GLOBAL-LATEST timestamp (404 @ 14:44) -> temp
    discarded, timestamp still sets the upper bound -> trailing blank.
  * Interior temp gaps filled by standard linear interpolation: the glitch
    cluster (Alpha 14:32/14:33), a JSON null (Alpha 14:36), and a no-row minute
    (Alpha 14:39); plus an empty Beta Fahrenheit field (Beta 14:33), a no-row
    minute (Beta 14:36), and a two-minute Beta gap (Beta 14:39/14:40).
  * Leading blank from Beta (no Beta row at 14:30); trailing blank from both
    sensors at 14:44.
  * Vibration is never interpolated; no-row Beta minutes leave beta_vibration_hz
    blank.
  * Beta has two readings at 14:32 (averaged) and assorted ISO-8601 offsets
    (Z, +02:00, -05:00) that must be normalized to UTC before bucketing.

This script is a dev-time generator. The Dockerfile should COPY only the two
produced files into /app/ -- not this script, and not any expected output.
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta, timezone

SEED = 20240512
random.seed(SEED)

# Whole-minute UTC anchor for the first timeline bucket (14:30:00 UTC).
BASE = datetime(2024, 5, 12, 14, 30, 0, tzinfo=timezone.utc)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")


def at(minute, second):
    """UTC datetime `minute`m`second`s into the window."""
    return BASE + timedelta(minutes=minute, seconds=second)


def epoch(dt):
    return int(dt.timestamp())


def ramp(minute, base, slope, jitter=0.3):
    """A gently rising sensor value with reproducible jitter, 1 decimal."""
    return round(base + slope * minute + random.uniform(-jitter, jitter), 1)


def iso_offset(dt_utc, offset_hours):
    """Render a UTC instant in a chosen integer ISO-8601 offset.

    14:31 UTC at +02:00 -> '...T16:31:20+02:00'; offset 0 -> trailing 'Z'.
    """
    tz = timezone(timedelta(hours=offset_hours))
    local = dt_utc.astimezone(tz)
    stamp = local.strftime("%Y-%m-%dT%H:%M:%S")
    if offset_hours == 0:
        return stamp + "Z"
    sign = "+" if offset_hours >= 0 else "-"
    return f"{stamp}{sign}{abs(offset_hours):02d}:00"


# --- Alpha rows: (utc_datetime, temp_celsius_or_None, status_code) -----------
# Hand-designed one-reading-per-minute sequence on whole-minute timestamps, so
# the 1.5 C/min thermal-inertia filter has clean, exact elapsed-minute math.
# Glitch spikes (35.0/36.0) must be dropped relative to the LAST VALID reading;
# the 404 row at the end is discarded by status but still sets the upper bound.
alpha = [
    (at(0, 0), 20.0, 200),    # 14:30 first reading -> accepted
    (at(1, 0), 20.5, 200),    # 14:31 0.5/min vs 14:30 -> valid (new last-valid)
    (at(2, 0), 35.0, 200),    # 14:32 14.5/min vs 14:31 -> GLITCH (drop temp)
    (at(3, 0), 36.0, 200),    # 14:33 7.75/min vs 14:31 -> GLITCH (drop temp)
    (at(4, 0), 22.0, 200),    # 14:34 1.5 C / 3 min vs 14:31 = 0.5/min -> valid
    (at(5, 0), 22.5, 200),    # 14:35 0.5/min -> valid
    (at(6, 0), None, 200),    # 14:36 null temp -> interpolate
    (at(7, 0), 23.5, 200),    # 14:37 1.0 C / 2 min vs 14:35 = 0.5/min -> valid
    (at(8, 0), 24.0, 200),    # 14:38 0.5/min -> valid
    # 14:39 no alpha row -> interpolate
    (at(10, 0), 25.0, 200),   # 14:40 1.0 C / 2 min vs 14:38 = 0.5/min -> valid
    (at(11, 0), 25.5, 200),   # 14:41 valid
    (at(12, 0), 26.0, 200),   # 14:42 valid
    (at(13, 0), 26.5, 200),   # 14:43 valid
    (at(14, 0), 99.0, 404),   # 14:44 non-200 -> temp discarded, sets upper bound -> trailing blank
]

# --- Beta rows: (utc_datetime, temp_fahrenheit_or_None, offset_hours) -------
beta = [
    (at(1, 20), ramp(1, 68.0, 0.9, 0.5), 2),
    (at(2, 5), ramp(2, 68.0, 0.9, 0.5), -5),     # 14:32 two readings (diff offsets) -> averaged
    (at(2, 50), ramp(2, 68.0, 0.9, 0.5), 0),
    (at(3, 30), None, 2),                        # 14:33 empty Fahrenheit -> interpolate
    (at(4, 10), ramp(4, 68.0, 0.9, 0.5), 0),
    (at(5, 40), ramp(5, 68.0, 0.9, 0.5), -5),
    # 14:36 no beta row -> temp interpolates, vibration blank
    (at(7, 15), ramp(7, 68.0, 0.9, 0.5), 2),
    (at(8, 25), ramp(8, 68.0, 0.9, 0.5), 0),
    # 14:39 + 14:40 no beta rows -> two-minute gap, standard multi-step interpolation
    (at(11, 50), ramp(11, 68.0, 0.9, 0.5), 2),
    (at(12, 20), ramp(12, 68.0, 0.9, 0.5), 0),
    (at(13, 10), ramp(13, 68.0, 0.9, 0.5), -5),
]

os.makedirs(OUT_DIR, exist_ok=True)

alpha_path = os.path.join(OUT_DIR, "sensor_alpha.log")
with open(alpha_path, "w") as f:
    for dt, temp, status in alpha:
        f.write(json.dumps({
            "timestamp": epoch(dt),
            "temp_celsius": temp,
            "status_code": status,
        }) + "\n")

beta_path = os.path.join(OUT_DIR, "sensor_beta.csv")
with open(beta_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["recorded_at", "temp_fahrenheit", "vibration_hz"])
    for dt, tempF, off in beta:
        vib = round(random.uniform(0.5, 5.0), 2)
        w.writerow([iso_offset(dt, off), "" if tempF is None else tempF, vib])

print(f"Wrote {alpha_path} ({len(alpha)} rows)")
print(f"Wrote {beta_path} ({len(beta)} rows)")
print(f"Timeline (UTC): {BASE:%H:%M}..{BASE + timedelta(minutes=14):%H:%M}  (15 one-minute buckets)")
print("Glitch cluster: 14:32/14:33 dropped vs last-valid 14:31; 14:34 kept (0.5 C/min over 3 min)")
