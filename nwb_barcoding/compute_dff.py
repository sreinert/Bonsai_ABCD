"""Compute suite2p dF/F ('dFF.npy') if it's missing, at the exact path Aimbot_barcoding expects.

Implements the CellTV-style green trace: neuropil-correct using the coefficient
stored in Suite2p's ops.npy, then apply Suite2p's own dcnv.preprocess baseline
correction. It is a plain function reading only .npy files, with no dependency on
the Bonsai_ABCD package itself.

Path convention matches exactly what PMPipelineSegmentationExtractor looks for
(Aimbot_barcoding's preprocess_alignment/nwb/ophys/extractors/pmpipeline_segmentation_extractor.py
::_load_npy): given `folder_path = <funcimg_dir>/suite2p` (what nwb_conversion_2p.py
passes as PMPipelineSegmentation's `folder_path`), it looks for dFF.npy at
`folder_path.with_name('dff') / <plane_name> / 'dFF.npy'`, i.e. `<funcimg_dir>/dff/<plane_name>/dFF.npy`
-- a *sibling* of `suite2p/`, not nested inside it. Note this differs from
extract_dff.ipynb's own current (GoalProgress-cohort) path variables, which drop the
`funcimg/` level entirely (`session_path/suite2p/plane0` and `session_path/dff/plane0`)
-- that mismatch is exactly why DfOverF was missing from NWBs built by nwb_barcoding's
predecessor pipeline for sessions where dFF.npy only exists under that other layout.
This module always writes to the layout the NWB conversion step actually reads from.
"""

from pathlib import Path

import numpy as np


def _suite2p_preprocess(fluorescence, baseline, win_baseline, sig_baseline, fs,
                        prctile_baseline):
    from suite2p.extraction import dcnv

    return dcnv.preprocess(
        fluorescence, baseline, win_baseline, sig_baseline, fs, prctile_baseline
    )


def _get_baseline_params(ops):
    """suite2p renamed these ops keys at some point ('dcnv_preprocess' nesting); support both."""
    try:
        return (ops['baseline'], ops['win_baseline'], ops['sig_baseline'], ops['prctile_baseline'])
    except KeyError:
        p = ops['dcnv_preprocess']
        return (p['baseline'], p['win_baseline'], p['sig_baseline'], p['prctile_baseline'])


def _get_neuropil_coefficient(ops, override=None):
    """Return Suite2p's stored neuropil coefficient unless explicitly overridden."""
    if override is not None:
        return float(override)
    if "neucoeff" in ops:
        return float(ops["neucoeff"])
    extraction = ops.get("extraction", {})
    if isinstance(extraction, dict) and "neuropil_coefficient" in extraction:
        return float(extraction["neuropil_coefficient"])
    raise KeyError(
        "ops.npy has no 'neucoeff' or extraction['neuropil_coefficient']; "
        "cannot reproduce Suite2p/CellTV neuropil correction"
    )


def compute_dff_for_plane(suite2p_plane_path, dff_plane_path, neuropil_coefficient=None):
    """Compute dFF for one plane from F.npy/Fneu.npy/ops.npy and save it as dFF.npy.

    Same algorithm as preprocessing/extract_dff.ipynb. Returns the path written.
    """
    suite2p_plane_path = Path(suite2p_plane_path)
    f = np.load(suite2p_plane_path / 'F.npy')
    fneu = np.load(suite2p_plane_path / 'Fneu.npy')
    ops = np.load(suite2p_plane_path / 'ops.npy', allow_pickle=True).item()

    neuropil_coefficient = _get_neuropil_coefficient(
        ops, override=neuropil_coefficient
    )
    f_corr = f - fneu * neuropil_coefficient
    baseline, win_baseline, sig_baseline, prctile_baseline = _get_baseline_params(ops)
    dff = _suite2p_preprocess(
        f_corr, baseline, win_baseline, sig_baseline, ops['fs'], prctile_baseline
    )

    dff_plane_path = Path(dff_plane_path)
    dff_plane_path.mkdir(parents=True, exist_ok=True)
    dff_file = dff_plane_path / 'dFF.npy'
    np.save(dff_file, dff)
    return dff_file


def ensure_dff(funcimg_dir, suite2p_dirname='suite2p', dff_dirname='dff', neuropil_coefficient=None,
                force=False):
    """Compute dFF.npy for every suite2p plane under `funcimg_dir` that's missing one.

    `funcimg_dir` is the folder containing `suite2p/` (i.e. the parent of the
    `folder_path` given to PMPipelineSegmentation in nwb_conversion_2p.py --
    normally `<preprocessed_session>/funcimg`). Safe to call unconditionally
    before building the funcimg NWB: does nothing for planes that already have
    a dFF.npy, unless `force=True`.

    Returns a list of dFF.npy paths (existing or newly computed), one per plane.
    """
    funcimg_dir = Path(funcimg_dir)
    suite2p_dir = funcimg_dir / suite2p_dirname
    plane_dirs = sorted(p for p in suite2p_dir.glob('plane*') if p.is_dir())
    if not plane_dirs:
        raise FileNotFoundError(f"No suite2p plane folders found under {suite2p_dir}")

    dff_paths = []
    for plane_dir in plane_dirs:
        plane_name = plane_dir.name
        dff_plane_path = funcimg_dir / dff_dirname / plane_name
        dff_file = dff_plane_path / 'dFF.npy'

        if dff_file.exists() and not force:
            print(f"dFF.npy already exists for {plane_name}, skipping: {dff_file}")
        elif not (plane_dir / 'F.npy').exists():
            print(f"Skipping {plane_name}: no F.npy found under {plane_dir} (nothing to compute dFF from).")
            continue
        else:
            coefficient_source = (
                f"explicit coefficient {neuropil_coefficient}"
                if neuropil_coefficient is not None
                else "the neuropil coefficient stored in ops.npy"
            )
            print(f"dFF.npy missing for {plane_name} -- computing it from {plane_dir} "
                  f"(F - neucoeff*Fneu using {coefficient_source}, then Suite2p "
                  "dcnv.preprocess baseline correction)...")
            dff_file = compute_dff_for_plane(plane_dir, dff_plane_path, neuropil_coefficient=neuropil_coefficient)
            print(f"Saved {dff_file}")
        dff_paths.append(dff_file)

    return dff_paths
