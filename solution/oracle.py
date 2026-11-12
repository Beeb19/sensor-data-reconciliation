#!/usr/bin/env python3
"""Oracle for corrupt-sensor-recon — builds aligned_report.csv from the two logs.

Implements instruction.md exactly. Runs unchanged in two contexts:

  * In the task container (Harbor oracle): reads /app/sensor_alpha.log and
    /app/sensor_beta.csv, writes /app/aligned_report.csv.
  * Locally: if the /app inputs are absent, falls back to environment/data/
    (resolved relative to this file, so it works from any working directory)
    and writes aligned_report.csv at the task root.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_RATE_C_PER_MIN = 1.5


def resolve_paths():
    """Prefer the in-container /app inputs; fall back to the local fixture."""
    app = Path("/app")
    if (app / "sensor_alpha.log").exists():
        return app / "sensor_alpha.log", app / "sensor_beta.csv", app / "aligned_report.csv"
    task_root = Path(__file__).resolve().parent.parent          # tasks/corrupt-sensor-recon
    data = task_root / "environment" / "data"
    return data / "sensor_alpha.log", data / "sensor_beta.csv", task_root / "aligned_report.csv"


def load_alpha(path):
    """Row-level (ts, temp_c). Non-200 status or null temp -> NaN; the timestamp
    is always kept so it counts toward the timeline bounds. The thermal-inertia
    glitch filter is applied afterwards by rate_filter, before bucketing."""
    timestamps, temps = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            timestamps.append(pd.to_datetime(row["timestamp"], unit="s", utc=True))
            if row.get("status_code") == 200 and row.get("temp_celsius") is not None:
                temps.append(float(row["temp_celsius"]))
            else:
                temps.append(np.nan)
    return pd.DataFrame({"ts": timestamps, "temp_c": temps})


def rate_filter(df):
    """Stateful thermal-inertia filter. Walking readings in time order, drop any
    reading whose absolute rate of change from the LAST ACCEPTED reading strictly
    exceeds MAX_RATE_C_PER_MIN. A dropped glitch does NOT become the reference for
    the next reading; the first real reading is always accepted. NaN rows
    (non-200 / null) are not readings — they pass through unchanged and do not
    update the reference. (A simple vectorised diff vs the previous row is WRONG:
    it would compare against discarded glitches.)"""
    df = df.sort_values("ts").reset_index(drop=True)
    last_ts = None
    last_temp = None
    kept = []
    for ts, temp in zip(df["ts"], df["temp_c"]):
        if pd.isna(temp):
            kept.append(np.nan)
            continue
        if last_temp is None:                       # first real reading: always accept
            kept.append(temp)
            last_ts, last_temp = ts, temp
            continue
        elapsed_min = (ts - last_ts).total_seconds() / 60.0
        rate = abs(temp - last_temp) / elapsed_min
        if rate > MAX_RATE_C_PER_MIN:               # glitch: drop temp, reference unchanged
            kept.append(np.nan)
        else:
            kept.append(temp)
            last_ts, last_temp = ts, temp
    df = df.copy()
    df["temp_c"] = kept
    df["minute"] = df["ts"].dt.floor("min")
    return df


def load_beta(path):
    """Floored-minute timestamps + temp_c (from Fahrenheit) + vibration.
    ISO-8601 offsets are normalized to UTC before flooring."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["recorded_at"], utc=True, format="ISO8601")
    temp_f = pd.to_numeric(df["temp_fahrenheit"], errors="coerce")   # empty field -> NaN
    out = pd.DataFrame({
        "ts": ts,
        "temp_c": (temp_f - 32.0) * 5.0 / 9.0,
        "vib": pd.to_numeric(df["vibration_hz"], errors="coerce"),
    })
    out["minute"] = out["ts"].dt.floor("min")
    return out


def main():
    alpha_path, beta_path, out_path = resolve_paths()
    alpha = rate_filter(load_alpha(alpha_path))   # status filter -> glitch filter -> minute
    beta = load_beta(beta_path)

    # Timeline: floor(earliest)..floor(latest) across BOTH logs. The min/max here
    # still include the discarded non-200 / glitch rows because their timestamps
    # are retained (only their temperatures were nulled).
    start = min(alpha["minute"].min(), beta["minute"].min())
    end = max(alpha["minute"].max(), beta["minute"].max())
    timeline = pd.date_range(start=start, end=end, freq="min")      # tz inherited (UTC)

    # Per-minute mean of the non-null readings in each bucket (mean skips NaN).
    alpha_t = alpha.groupby("minute")["temp_c"].mean().reindex(timeline)
    beta_t = beta.groupby("minute")["temp_c"].mean().reindex(timeline)
    beta_v = beta.groupby("minute")["vib"].mean().reindex(timeline)

    # Standard linear interpolation of temperatures only; limit_area="inside"
    # leaves leading/trailing gaps blank. Vibration is never interpolated.
    alpha_t = alpha_t.interpolate(method="linear", limit_area="inside")
    beta_t = beta_t.interpolate(method="linear", limit_area="inside")

    report = pd.DataFrame({
        "timestamp_utc": timeline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alpha_temp_c": alpha_t.round(2).to_numpy(),
        "beta_temp_c": beta_t.round(2).to_numpy(),
        "beta_vibration_hz": beta_v.round(2).to_numpy(),
    })
    report.to_csv(out_path, index=False)        # NaN -> empty field (default na_rep)
    print(f"Wrote {out_path} ({len(report)} rows)")


if __name__ == "__main__":
    main()
