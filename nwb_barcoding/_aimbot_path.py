"""Makes Aimbot_barcoding's `preprocess_alignment` package importable.

This pipeline deliberately does not vendor or reimplement Aimbot_barcoding's
NWB-writing machinery (neuroconv converters/interfaces for TTL tables, suite2p
ophys segmentation, etc.) -- that code already works and duplicating it would
just be a second copy to keep in sync. Instead we import it directly, the same
way you'd depend on any sibling library, and only add/override the specific
pieces that needed fixing (see robust_converter.py).

Must run in the `2p_preprocess_extraction` conda env (same one Aimbot_barcoding
itself requires) -- pynwb==3.0.0, neuroconv==0.7.2, forked ndx-events/roiextractors.
Not compatible with this repo's own `bonsai_abcd` env.
"""

import sys
from pathlib import Path

# Sibling repo, same parent directory as this one (Bonsai_ABCD).
AIMBOT_BARCODING_ROOT = Path(__file__).resolve().parents[2] / "Aimbot_barcoding"


def ensure_on_path():
    if not AIMBOT_BARCODING_ROOT.exists():
        raise FileNotFoundError(
            f"Aimbot_barcoding repo not found at {AIMBOT_BARCODING_ROOT}. "
            "This pipeline expects it as a sibling of Bonsai_ABCD."
        )
    root_str = str(AIMBOT_BARCODING_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return AIMBOT_BARCODING_ROOT
