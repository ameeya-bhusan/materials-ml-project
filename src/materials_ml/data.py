"""Data ingestion for Materials Project Li insertion electrodes.

Pulls insertion-electrode records from the Materials Project API,
flattens them into a tidy DataFrame, and applies physics-based cleaning.
"""

from __future__ import annotations

import pandas as pd
from mp_api.client import MPRester

# Fields we request from the API. Kept as a module constant so the
# notebook, the function, and any test all agree on one source of truth.
ELECTRODE_FIELDS = [
    "average_voltage",
    "battery_formula", "framework_formula",
    "formula_charge", "formula_discharge",
    "working_ion", "elements", "nelements", "chemsys",
    "capacity_grav", "capacity_vol",
    "energy_grav", "energy_vol",
    "max_delta_volume", "num_steps",
    "stability_charge", "stability_discharge",
    "battery_id", "id_charge", "id_discharge", "warnings",
]


def fetch_electrodes(working_ion: str = "Li") -> pd.DataFrame:
    """Fetch insertion-electrode records and flatten to a DataFrame.

    Parameters
    ----------
    working_ion
        The shuttling ion to filter on (e.g. "Li", "Na", "Mg").

    Returns
    -------
    DataFrame with one row per electrode, object fields flattened
    to plain strings/lists so the result is Parquet-serializable.
    """
    with MPRester() as mpr:
        docs = mpr.materials.insertion_electrodes.search(
            working_ion=working_ion,
            fields=ELECTRODE_FIELDS,
        )

    rows = [
        {
            "battery_id": doc.battery_id,
            "average_voltage": doc.average_voltage,
            "battery_formula": doc.battery_formula,
            "framework_formula": doc.framework_formula,
            "formula_charge": doc.formula_charge,
            "formula_discharge": doc.formula_discharge,
            "working_ion": str(doc.working_ion),
            "elements": [str(e) for e in doc.elements],
            "nelements": doc.nelements,
            "chemsys": doc.chemsys,
            "capacity_grav": doc.capacity_grav,
            "capacity_vol": doc.capacity_vol,
            "energy_grav": doc.energy_grav,
            "energy_vol": doc.energy_vol,
            "max_delta_volume": doc.max_delta_volume,
            "num_steps": doc.num_steps,
            "stability_charge": doc.stability_charge,
            "stability_discharge": doc.stability_discharge,
            "id_charge": str(doc.id_charge),
            "id_discharge": str(doc.id_discharge),
            "n_warnings": len(doc.warnings) if doc.warnings else 0,
            "warnings": "; ".join(doc.warnings) if doc.warnings else "",
        }
        for doc in docs
    ]
    return pd.DataFrame(rows)


def clean_electrodes(
    df: pd.DataFrame,
    v_min: float = 0.0,
    v_max: float = 5.5,
) -> pd.DataFrame:
    """Filter to physically viable Li cathode voltages.

    A non-positive insertion voltage means the lithiated phase is not a
    viable cathode; voltages above ~5.5 V exceed real Li-cathode chemistry
    and are computational artifacts. Warnings are intentionally NOT used
    as a filter (they were found not to correlate with voltage quality)
    and are kept as features.

    Parameters
    ----------
    df
        Raw electrode DataFrame from `fetch_electrodes`.
    v_min, v_max
        Inclusive-exclusive voltage bounds in volts.

    Returns
    -------
    Cleaned DataFrame with a reset index.
    """
    mask = (df["average_voltage"] > v_min) & (df["average_voltage"] <= v_max)
    return df[mask].copy().reset_index(drop=True)