import os
import pandas as pd
import pytest

AGENT_FILE = "/app/aligned_report.csv"
EXPECTED_FILE = os.path.join(os.path.dirname(__file__), "expected_report.csv")
VALUE_COLS = ["alpha_temp_c", "beta_temp_c", "beta_vibration_hz"]
EXPECTED_COLUMNS = ["timestamp_utc"] + VALUE_COLS

@pytest.fixture(scope="module")
def frames():
    assert os.path.exists(AGENT_FILE), f"agent output not found at {AGENT_FILE}"
    assert os.path.exists(EXPECTED_FILE), f"expected report missing at {EXPECTED_FILE}"
    return pd.read_csv(AGENT_FILE), pd.read_csv(EXPECTED_FILE)

def test_columns_exact_order(frames):
    """instruction.md: exact headers in order — timestamp_utc, alpha_temp_c, beta_temp_c, beta_vibration_hz."""
    agent_df, _ = frames
    assert list(agent_df.columns) == EXPECTED_COLUMNS

def test_timeline_matches(frames):
    """instruction.md: 1-minute UTC grid floor(earliest)..floor(latest), each row %Y-%m-%dT%H:%M:%SZ — exact strings, order, and bounds."""
    agent_df, expected_df = frames
    assert agent_df["timestamp_utc"].equals(expected_df["timestamp_utc"])

def test_values_within_tolerance(frames):
    """instruction.md: F->C, average non-null, linear interpolation, 2-dp rounding, blanks where un-interpolatable."""
    agent_df, expected_df = frames
    pd.testing.assert_frame_equal(
        agent_df[VALUE_COLS],
        expected_df[VALUE_COLS],
        check_dtype=False,
        check_exact=False,
        atol=0.02,
    )
