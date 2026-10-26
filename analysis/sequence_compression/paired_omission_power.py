"""Power calculations for paired omitted-A versus B responses within a mouse.

Each omitted-A observation must be matched to one comparable B observation from
the same mouse and, normally, the same session.  The calculation is therefore
for a paired t test on ``omitted A - B`` differences.  It is appropriate for a
continuous response such as lick count, lick rate, speed, or fluorescence when
the paired differences are reasonably well behaved.

Example
-------
The input CSV can be in long form::

    mouse,session,pair_id,condition,response
    01,001,0,omitted_A,5
    01,001,0,B,2
    01,001,1,omitted_A,4
    01,001,1,B,3

Run::

    python paired_omission_power.py pilot.csv --exclusion-rate 0.2

The reported ``required_valid_pairs`` is the number of usable omitted-A/B
pairs. ``planned_probe_trials`` inflates that number for the expected exclusion
rate.  Since one pair contains one omitted A, this is also the number of
omitted-A probes to plan for each mouse.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import nct, t


def paired_effect_size(a: Iterable[float], b: Iterable[float]) -> float:
    """Return Cohen's dz for matched arrays, defined as mean(A-B)/SD(A-B)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must contain the same number of matched values")

    differences = a - b
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return np.nan

    sd = np.std(differences, ddof=1)
    if sd == 0:
        return np.inf if np.mean(differences) != 0 else np.nan
    return float(np.mean(differences) / sd)


def paired_t_power(
    effect_size: float,
    n_pairs: int,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Return exact noncentral-t power for a paired t test."""
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if not np.isfinite(effect_size):
        return 1.0 if np.isinf(effect_size) else np.nan

    df = n_pairs - 1
    noncentrality = effect_size * math.sqrt(n_pairs)
    if alternative == "two-sided":
        critical = t.ppf(1 - alpha / 2, df)
        return float(
            nct.cdf(-critical, df, noncentrality)
            + nct.sf(critical, df, noncentrality)
        )
    if alternative == "greater":
        critical = t.ppf(1 - alpha, df)
        return float(nct.sf(critical, df, noncentrality))

    critical = t.ppf(alpha, df)
    return float(nct.cdf(critical, df, noncentrality))


def required_pairs(
    effect_size: float,
    *,
    target_power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    max_pairs: int = 10_000_000,
) -> int | None:
    """Return the smallest number of valid pairs attaining ``target_power``.

    For a two-sided test, only the magnitude of the effect matters.  For a
    directional test, the sign must agree with the requested alternative.
    ``None`` is returned when the requested power is not reached.
    """
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between 0 and 1")
    if max_pairs < 2:
        raise ValueError("max_pairs must be at least 2")
    if not np.isfinite(effect_size):
        return 2 if np.isinf(effect_size) else None

    effect = abs(effect_size) if alternative == "two-sided" else effect_size
    if effect == 0:
        return None
    if alternative == "greater" and effect < 0:
        return None
    if alternative == "less" and effect > 0:
        return None

    def reaches_target(n_pairs: int) -> bool:
        return paired_t_power(
            effect,
            n_pairs,
            alpha=alpha,
            alternative=alternative,
        ) >= target_power

    if reaches_target(2):
        return 2

    # Find an upper bound quickly, then locate the smallest valid sample size.
    # This matters for very small raw effects such as 0.05 licks/landmark.
    low, high = 2, 4
    while high < max_pairs and not reaches_target(high):
        low = high
        high = min(high * 2, max_pairs)
    if not reaches_target(high):
        return None

    while low + 1 < high:
        midpoint = (low + high) // 2
        if reaches_target(midpoint):
            high = midpoint
        else:
            low = midpoint
    return high


def make_paired_table(
    data: pd.DataFrame,
    *,
    mouse_col: str = "mouse",
    pair_cols: Sequence[str] = ("session", "pair_id"),
    condition_col: str = "condition",
    response_col: str = "response",
    a_label: str = "omitted_A",
    b_label: str = "B",
) -> pd.DataFrame:
    """Validate long-form observations and return one row per matched pair."""
    index_cols = [mouse_col, *pair_cols]
    required = set(index_cols + [condition_col, response_col])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    selected = data[data[condition_col].isin([a_label, b_label])].copy()
    duplicates = selected.duplicated(index_cols + [condition_col], keep=False)
    if duplicates.any():
        example = selected.loc[duplicates, index_cols + [condition_col]].iloc[0]
        raise ValueError(
            "each pair must contain at most one observation per condition; "
            f"duplicate found at {example.to_dict()}"
        )

    paired = selected.pivot(
        index=index_cols,
        columns=condition_col,
        values=response_col,
    ).reset_index()
    paired.columns.name = None
    for label in (a_label, b_label):
        if label not in paired:
            paired[label] = np.nan

    paired = paired.dropna(subset=[a_label, b_label]).copy()
    paired["difference_A_minus_B"] = paired[a_label] - paired[b_label]
    return paired


def analyse_by_mouse(
    paired: pd.DataFrame,
    *,
    mouse_col: str = "mouse",
    a_label: str = "omitted_A",
    b_label: str = "B",
    minimum_difference: float | None = None,
    target_power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    exclusion_rate: float = 0.2,
) -> pd.DataFrame:
    """Estimate effect and required omitted-A probes separately for each mouse.

    If ``minimum_difference`` is supplied, power is based on that scientifically
    meaningful raw A-minus-B difference and the pilot SD of paired differences.
    Otherwise it is based on the observed pilot mean difference.
    """
    if not 0 <= exclusion_rate < 1:
        raise ValueError("exclusion_rate must be in [0, 1)")

    rows = []
    for mouse, group in paired.groupby(mouse_col, sort=False, dropna=False):
        differences = group["difference_A_minus_B"].to_numpy(dtype=float)
        n_pilot = len(differences)
        mean_difference = float(np.mean(differences)) if n_pilot else np.nan
        sd_difference = (
            float(np.std(differences, ddof=1)) if n_pilot >= 2 else np.nan
        )
        observed_dz = paired_effect_size(group[a_label], group[b_label])

        powered_difference = (
            float(minimum_difference)
            if minimum_difference is not None
            else mean_difference
        )
        if np.isfinite(sd_difference) and sd_difference > 0:
            powered_dz = powered_difference / sd_difference
        elif sd_difference == 0 and powered_difference != 0:
            powered_dz = math.copysign(np.inf, powered_difference)
        else:
            powered_dz = np.nan

        n_required = required_pairs(
            powered_dz,
            target_power=target_power,
            alpha=alpha,
            alternative=alternative,
        )
        n_planned = (
            math.ceil(n_required / (1 - exclusion_rate))
            if n_required is not None
            else None
        )
        rows.append(
            {
                mouse_col: mouse,
                "pilot_pairs": n_pilot,
                "mean_omitted_A": float(group[a_label].mean()),
                "mean_B": float(group[b_label].mean()),
                "mean_difference_A_minus_B": mean_difference,
                "sd_paired_difference": sd_difference,
                "observed_dz": observed_dz,
                "powered_difference": powered_difference,
                "powered_dz": powered_dz,
                "required_valid_pairs": n_required,
                "planned_probe_trials": n_planned,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(
    effect_sizes: Sequence[float] = (0.2, 0.3, 0.5, 0.8, 1.0),
    *,
    target_power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    exclusion_rate: float = 0.2,
) -> pd.DataFrame:
    """Return required pairs over a range of plausible standardized effects."""
    rows = []
    for effect_size in effect_sizes:
        signed_effect = (
            -abs(effect_size) if alternative == "less" else abs(effect_size)
        )
        n_required = required_pairs(
            signed_effect,
            target_power=target_power,
            alpha=alpha,
            alternative=alternative,
        )
        rows.append(
            {
                "dz": effect_size,
                "required_valid_pairs": n_required,
                "planned_probe_trials": math.ceil(
                    n_required / (1 - exclusion_rate)
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Long-form pilot observations")
    parser.add_argument("--mouse-col", default="mouse")
    parser.add_argument("--pair-cols", nargs="+", default=["session", "pair_id"])
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--response-col", default="response")
    parser.add_argument("--a-label", default="omitted_A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--alternative", choices=["two-sided", "greater", "less"], default="two-sided"
    )
    parser.add_argument("--exclusion-rate", type=float, default=0.2)
    parser.add_argument(
        "--minimum-difference",
        type=float,
        help="Smallest meaningful raw A-minus-B difference; defaults to pilot mean",
    )
    parser.add_argument("--output", type=Path, help="Optional output CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = pd.read_csv(args.csv)
    paired = make_paired_table(
        data,
        mouse_col=args.mouse_col,
        pair_cols=args.pair_cols,
        condition_col=args.condition_col,
        response_col=args.response_col,
        a_label=args.a_label,
        b_label=args.b_label,
    )
    result = analyse_by_mouse(
        paired,
        mouse_col=args.mouse_col,
        a_label=args.a_label,
        b_label=args.b_label,
        minimum_difference=args.minimum_difference,
        target_power=args.power,
        alpha=args.alpha,
        alternative=args.alternative,
        exclusion_rate=args.exclusion_rate,
    )
    print("Per-mouse pilot estimate and power calculation")
    print(result.to_string(index=False))
    print("\nSensitivity analysis")
    print(
        sensitivity_table(
            target_power=args.power,
            alpha=args.alpha,
            alternative=args.alternative,
            exclusion_rate=args.exclusion_rate,
        ).to_string(index=False)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
