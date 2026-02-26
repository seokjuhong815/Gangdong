"""Two-point variable flip-angle (VFA) T1 mapping.

Implements the linearized spoiled gradient echo relation using two flip-angle images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compute_t1_map_two_flip_angles(
    signal_fa1: np.ndarray,
    signal_fa2: np.ndarray,
    fa1_deg: float = 5.0,
    fa2_deg: float = 15.0,
    tr_ms: float = 10.0,
    te_ms: float = 1.93333333,
    min_t1_ms: float = 1.0,
    max_t1_ms: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute T1 and M0 maps from two spoiled-GRE images.

    Parameters
    ----------
    signal_fa1, signal_fa2:
        Same-shape arrays acquired at ``fa1_deg`` and ``fa2_deg``.
    fa1_deg, fa2_deg:
        Flip angles in degrees.
    tr_ms:
        Repetition time in milliseconds.
    te_ms:
        Echo time in milliseconds. Kept for metadata/reporting; not used by this model.
    min_t1_ms, max_t1_ms:
        Clipping range for physically meaningful output.

    Returns
    -------
    t1_ms, m0, e1:
        T1 map [ms], proton-density-like M0, and E1=exp(-TR/T1).
    """

    del te_ms  # TE is not part of the two-point VFA signal model itself.

    s1 = np.asarray(signal_fa1, dtype=np.float64)
    s2 = np.asarray(signal_fa2, dtype=np.float64)
    if s1.shape != s2.shape:
        raise ValueError(f"Shape mismatch: {s1.shape=} vs {s2.shape=}")

    a1 = np.deg2rad(fa1_deg)
    a2 = np.deg2rad(fa2_deg)

    # Linearization of spoiled-GRE equation:
    # y = S/sin(a), x = S/tan(a), y = E1*x + M0*(1-E1)
    x1 = s1 / np.tan(a1)
    y1 = s1 / np.sin(a1)
    x2 = s2 / np.tan(a2)
    y2 = s2 / np.sin(a2)

    denom = x2 - x1
    numer = y2 - y1

    e1 = np.full_like(s1, np.nan, dtype=np.float64)
    valid = np.isfinite(denom) & np.isfinite(numer) & (np.abs(denom) > 1e-12)
    e1[valid] = numer[valid] / denom[valid]

    # E1 must be in (0,1) for positive finite T1.
    e1_valid = (e1 > 0.0) & (e1 < 1.0)
    t1_ms = np.full_like(e1, np.nan, dtype=np.float64)
    t1_ms[e1_valid] = -tr_ms / np.log(e1[e1_valid])

    # Optional clipping to reduce outlier impact.
    t1_ms = np.where((t1_ms >= min_t1_ms) & (t1_ms <= max_t1_ms), t1_ms, np.nan)

    # Rearranged intercept to recover M0.
    m0 = np.full_like(e1, np.nan, dtype=np.float64)
    m0_denom = 1.0 - e1
    m0_valid = e1_valid & (np.abs(m0_denom) > 1e-12)
    m0[m0_valid] = y1[m0_valid] / m0_denom[m0_valid]

    return t1_ms, m0, e1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute two-point VFA T1 map from NumPy .npy volumes.",
    )
    parser.add_argument("fa5_npy", type=Path, help="Signal image (.npy) at 5 degrees")
    parser.add_argument("fa15_npy", type=Path, help="Signal image (.npy) at 15 degrees")
    parser.add_argument("--tr-ms", type=float, default=10.0, help="TR in ms (default: 10)")
    parser.add_argument(
        "--te-ms",
        type=float,
        default=1.93333333,
        help="TE in ms, not used in the model (default: 1.93333333)",
    )
    parser.add_argument("--fa1-deg", type=float, default=5.0, help="Flip angle 1 in degrees")
    parser.add_argument("--fa2-deg", type=float, default=15.0, help="Flip angle 2 in degrees")
    parser.add_argument("--out-t1", type=Path, default=Path("t1_map_ms.npy"))
    parser.add_argument("--out-m0", type=Path, default=Path("m0_map.npy"))
    parser.add_argument("--out-e1", type=Path, default=Path("e1_map.npy"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    s1 = np.load(args.fa5_npy)
    s2 = np.load(args.fa15_npy)

    t1_ms, m0, e1 = compute_t1_map_two_flip_angles(
        s1,
        s2,
        fa1_deg=args.fa1_deg,
        fa2_deg=args.fa2_deg,
        tr_ms=args.tr_ms,
        te_ms=args.te_ms,
    )

    np.save(args.out_t1, t1_ms)
    np.save(args.out_m0, m0)
    np.save(args.out_e1, e1)

    finite_count = np.isfinite(t1_ms).sum()
    print(f"Saved T1 map to {args.out_t1} (finite voxels: {finite_count}/{t1_ms.size})")


if __name__ == "__main__":
    main()
