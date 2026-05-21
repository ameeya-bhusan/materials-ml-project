"""Tests for the data cleaning logic."""

import pandas as pd

from materials_ml.data import clean_electrodes


def _toy_df():
    """A tiny synthetic DataFrame spanning the edge cases."""
    return pd.DataFrame({
        "average_voltage": [-2.0, 0.0, 0.5, 3.5, 5.5, 8.0],
        "battery_id": ["a", "b", "c", "d", "e", "f"],
    })


def test_clean_drops_nonpositive_and_high_voltages():
    """Entries <= 0 V or > 5.5 V are removed; valid ones kept."""
    out = clean_electrodes(_toy_df())
    # Keeps 0.5, 3.5, 5.5 (>0 and <=5.5); drops -2.0, 0.0, 8.0
    assert list(out["average_voltage"]) == [0.5, 3.5, 5.5]


def test_clean_respects_custom_bounds():
    """Custom thresholds are honored, proving they're real parameters."""
    out = clean_electrodes(_toy_df(), v_min=1.0, v_max=4.0)
    assert list(out["average_voltage"]) == [3.5]


def test_clean_resets_index():
    """Returned index is clean 0..n-1, not the original sparse index."""
    out = clean_electrodes(_toy_df())
    assert list(out.index) == [0, 1, 2]