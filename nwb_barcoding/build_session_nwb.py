"""Build behav + funcimg + merged NWB files for one session, with robust barcode alignment.

Usage (must run in the `2p_preprocess_extraction` conda env -- same one
Aimbot_barcoding itself requires; this repo's own `bonsai_abcd` env does not
have pynwb/neuroconv/etc. installed):

    # GoalProgress-style (single-tree) cohort:
    python -m nwb_barcoding.build_session_nwb sub-002 t2 \\
        "/Volumes/mrsic_flogel/public/projects/SaReMaRa_20260629_GoalProgress/rawdata" \\
        "/Volumes/mrsic_flogel/public/projects/SaReMaRa_20260629_GoalProgress/preprocessed" \\
        "/Volumes/mrsic_flogel/public/projects/SaReMaRa_20260629_GoalProgress/BehaviourAligned_robust"

    # cohort3-style (dual-tree) cohort:
    python -m nwb_barcoding.build_session_nwb sub-001 1LM014 \\
        "/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/rawdata" \\
        "/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives" \\
        "/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/BehaviourAligned_robust"

This reuses Aimbot_barcoding's (unmodified) convert_2p_to_nwb/convert_vrabcd_to_nwb
for the two individual NWBs -- that machinery already works and isn't touched.
What's different:
  - path resolution is cohort-agnostic (see build_session()'s docstring) and
    always reads behavior data from the true `rawdata/` tree;
  - output goes to a NEW, isolated tree instead of overwriting the existing
    behav_aligned/BehaviourAligned/preprocessed_behav_Nov2025 output, so real
    pipeline outputs are never at risk;
  - dFF.npy is computed automatically (compute_dff.py) if missing, before the
    funcimg NWB is built, so DfOverF is never silently absent;
  - the merged NWB is built with the corrected barcode alignment from
    barcode_alignment.py + robust_converter.py instead of Aimbot's
    intercept-only align_barcode_between_nwb; and
  - verify_session.py's checks run automatically against the result.
"""

from pathlib import Path

from . import _aimbot_path
_aimbot_path.ensure_on_path()

from preprocess_alignment.nwb_conversion_2p import convert_2p_to_nwb
from preprocess_alignment.nwb_conversion_vrabcd import convert_vrabcd_to_nwb
from neuroconv.utils import load_dict_from_file
import preprocess_alignment.nwb.behavior.Bonsai_ABCD.parse_bonsai_functions as parse_bonsai_functions
import os
import shutil
from pynwb import NWBHDF5IO

from .robust_converter import RobustVRABCDConverter
from . import barcode_alignment as ba
from . import verify_session
from . import compute_dff
from . import build_behaviour_data
from . import verify_behaviour_data


def _resolve_session_from_tag(subject_dir: Path, session_tag: str) -> str:
    """Match a session folder by tag, supporting both naming conventions seen across cohorts:
    GoalProgress-style suffix tags (folder ends with "-t2") and cohort3-style tags that
    appear as a substring anywhere in the name (e.g. "1LM014" in "ses-1LM014_date-...",
    matching parse_nwb_functions.find_base_path's `session in folder.name` convention).
    Suffix matching is tried first since it's less prone to accidental collisions
    (e.g. tag "t1" as a substring of "t10").
    """
    suffix_tag = session_tag if session_tag.startswith("-") else f"-{session_tag}"
    suffix_matches = sorted([p.name for p in subject_dir.iterdir() if p.is_dir() and p.name.endswith(suffix_tag)])
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"Multiple session folders match tag '{suffix_tag}' under {subject_dir}: {suffix_matches}")

    substring_matches = sorted([p.name for p in subject_dir.iterdir() if p.is_dir() and session_tag in p.name])
    if len(substring_matches) == 0:
        raise FileNotFoundError(f"No session folder found under {subject_dir} matching tag '{session_tag}'.")
    if len(substring_matches) > 1:
        # Same ambiguity parse_nwb_functions.find_base_path silently resolves elsewhere in this
        # codebase (e.g. cohort3 sometimes has two funcimg-preprocessing folders for one session,
        # from a rerun) by taking the last folder iterated. We do the same, but say so.
        print(f"WARNING: {len(substring_matches)} session folders contain tag '{session_tag}' under "
              f"{subject_dir}: {substring_matches} -- using the last one: '{substring_matches[-1]}'")
    return substring_matches[-1]


def build_merged_nwb_robust(preprocessed_funcimg_path: str, preprocessed_behav_path: str,
                             rawdata_path: str, nidaq_barcode_height: float = 0.5,
                             vdaq_barcode_height: float = 0.5, nidaq_nominal_rate: float = 9000.0):
    """Drop-in replacement for Aimbot_barcoding's align_barcode_between_nwb (vrabcd only).

    Same disk-level strategy as the original (copy the funcimg NWB, then append
    time-shifted behavior data into the copy) and same argument meaning, but
    computes (delta_t, corrected_nidaq_rate) via barcode_alignment.fit_alignment()
    instead of Aimbot's intercept-only calculate_temporal_difference(), and uses
    RobustVRABCDConverter to actually apply the corrected rate during conversion.

    Returns (merged_nwb_path, fit) -- `fit` is barcode_alignment.fit_alignment()'s
    return dict, reused by verify_session so its cross-check doesn't redo the fit.
    """
    preprocessed_funcimg_path = Path(preprocessed_funcimg_path)
    preprocessed_behav_path = Path(preprocessed_behav_path)
    funcimg_path = preprocessed_funcimg_path / 'funcimg'
    behav_path = preprocessed_behav_path / 'behav'

    nwbfile_path_2p = list(funcimg_path.glob("2p_preprocessed_*.nwb"))[0]
    if preprocessed_funcimg_path != preprocessed_behav_path:
        _new_path = preprocessed_behav_path / 'funcimg' / nwbfile_path_2p.name
        (preprocessed_behav_path / 'funcimg').mkdir(parents=True, exist_ok=True)
        shutil.move(str(nwbfile_path_2p), str(_new_path))
        nwbfile_path_2p = _new_path

    merged_nwb_filename = "vrabcd_" + Path(nwbfile_path_2p).name
    nwbfile_path_merged = str(preprocessed_behav_path / merged_nwb_filename)
    # Build into a "partial_" file (doesn't match the vrabcd_2p_preprocessed_* glob) and rename only
    # once the behaviour append succeeded -- previously a crash mid-append left a finished-looking
    # merged NWB with no behaviour in it, which every rerun then skipped. And a real copy, not
    # Aimbot's copy_nwb: that hard-links when it can, so appending into the merged file silently
    # appended into the funcimg NWB too.
    nwbfile_path_partial = str(preprocessed_behav_path / ("partial_" + merged_nwb_filename))
    if Path(nwbfile_path_partial).exists():
        Path(nwbfile_path_partial).unlink()
    shutil.copyfile(str(nwbfile_path_2p), nwbfile_path_partial)

    matches = list(behav_path.glob("vrabcd_preprocessed_*.nwb"))
    if not matches:
        raise FileNotFoundError(f"No behavioral NWB file found in {behav_path}")
    nwbfile_path_behav = matches[0]

    print("Decoding barcodes from funcimg (main) and behav (secondary) NWBs...")
    barcode_funcimg = ba.read_syncttl(nwbfile_path_2p, height=vdaq_barcode_height)
    barcode_behav = ba.read_syncttl(nwbfile_path_behav, height=nidaq_barcode_height)
    print(f"  funcimg: decoded {len(barcode_funcimg['values'])} barcodes at {barcode_funcimg['sample_rate']:.1f} Hz")
    print(f"  behav:   decoded {len(barcode_behav['values'])} barcodes (nominal rate {nidaq_nominal_rate} Hz)")

    fit = ba.fit_alignment(main_decoded=barcode_funcimg, secondary_decoded=barcode_behav,
                            secondary_nominal_rate=nidaq_nominal_rate)
    match = fit['match']
    print(f"  {len(match['shared_values'])} shared barcodes, "
          f"mean|residual|={abs(match['residual_ms']).mean():.2f} ms, "
          f"max|residual|={abs(match['residual_ms']).max():.2f} ms, "
          f"{match['n_outliers_excluded']} outlier(s) excluded")
    ppm_off = 1e6 * (fit['corrected_secondary_rate'] / nidaq_nominal_rate - 1)
    print(f"  delta_t = {fit['delta_t']:.6f} s, corrected NIDAQ rate = {fit['corrected_secondary_rate']:.4f} Hz "
          f"(nominal {nidaq_nominal_rate} Hz, {ppm_off:.1f} ppm off)")

    # Rebuilds the same source_data/metadata as convert_vrabcd_to_nwb's align_barcode=True
    # branch (nwb_conversion_vrabcd.py), but with RobustVRABCDConverter so the corrected
    # rate can be injected before conversion runs.
    ses_settings, ses_rig_settings = parse_bonsai_functions.load_settings(Path(rawdata_path))
    source_data = dict(AeonPandasData=dict(file_path=str(behav_path / "parsed_bonsai_data.pkl")))
    if ses_rig_settings["lickDetectorPort"] == 'COM4':
        source_data.update(dict(NIDAQData=dict(
            file_path=str(rawdata_path),
            ttl_stream_channels_to_names_map=dict(
                Dev2ai3='Photodiode1ONTTL', Dev2ai4='Photodiode2ONTTL', Dev2ai13='CameraFrameTTL',
                Dev2ai6='MODD1ONTTL', Dev2ai14='MODD2ONTTL',
            ),
        )))

    behavior_meta_path = (_aimbot_path.AIMBOT_BARCODING_ROOT / 'preprocess_alignment'
                           / 'nwb' / 'metadata' / 'behav_metadata_vrabcd.yml')
    behavior_metadata = load_dict_from_file(behavior_meta_path)

    converter = RobustVRABCDConverter(source_data=source_data, behavior_metadata=behavior_metadata, verbose=False)
    converter.set_corrected_rates(nidaq_rate=fit['corrected_secondary_rate'])
    # metadata intentionally not passed here -- matches nwb_conversion_vrabcd.py's own
    # align_barcode=True branch, which lets run_conversion() recompute it internally.
    converter.run_conversion(
        nwbfile_path=nwbfile_path_partial, conversion_options=dict(), overwrite=False,
        align_barcode=True, align_barcode_deltat=fit['delta_t'],
    )
    if not merged_nwb_is_complete(nwbfile_path_partial):
        raise RuntimeError(f"Behaviour append finished but {nwbfile_path_partial} has no NIDAQ TTL tables.")
    os.replace(nwbfile_path_partial, nwbfile_path_merged)

    return nwbfile_path_merged, fit


def merged_nwb_is_complete(nwb_path):
    """True if the merged NWB actually contains the appended behaviour (NIDAQ TTL tables), not
    just the copied funcimg content."""
    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        acquisition = io.read().acquisition
        return 'NIDAQ_TTLsTable' in acquisition and 'NIDAQ_TTLTypesTable' in acquisition


def build_session(subject: str, session_tag: str, rawdata_root: str, funcimg_preprocessed_root: str,
                   output_root: str, dff_neuropil_coefficient: float = 0.7,
                   build_behaviour_data_files: bool = True, force: bool = False):
    """Build behav/funcimg/merged NWBs for one session and verify the result.

    Cohort-agnostic: works for both the GoalProgress-style single-tree layout
    (rawdata/preprocessed/BehaviourAligned all share one session-folder name)
    and the cohort3-style dual-tree layout (rawdata, preprocessed_behav_Nov2025,
    and preprocessed_funcimg_Nov2025 each use their OWN session-folder naming
    for the same session, e.g. "ses-full030" vs "ses-005_id-full030_date-...").
    Behavior data is always read from `rawdata_root` (the one place guaranteed
    to have raw rig-settings/session-settings/CSV streams for both cohorts --
    cohort3's preprocessed_behav_Nov2025 output only contains already-parsed
    results, not raw CSVs, unlike GoalProgress's preprocessed/ mirror).

    :param str subject: e.g. "sub-002"
    :param str session_tag: session-folder suffix to match, e.g. "t2"
    :param str rawdata_root: the project's `rawdata/` folder, e.g.
        ".../SaReMaRa_20260629_GoalProgress/rawdata" or
        ".../SuKuSaRe_20250923_HFScohort3/rawdata"
    :param str funcimg_preprocessed_root: folder containing sub-XXX/<funcimg-session>/funcimg/suite2p --
        for GoalProgress-style cohorts this is ".../preprocessed"; for cohort3-style
        cohorts this is ".../preprocessed_funcimg_Nov2025/derivatives"
    :param str output_root: where to write the new behav/funcimg/merged NWBs, e.g.
        ".../SaReMaRa_20260629_GoalProgress/BehaviourAligned_robust" -- a NEW,
        isolated tree, never the existing behav_aligned/BehaviourAligned output.
    :param float dff_neuropil_coefficient: neuropil correction factor used if
        dFF.npy needs to be computed (see compute_dff.py); 0.7 matches
        preprocessing/extract_dff.ipynb's default.
    :param bool build_behaviour_data_files: if True (default), also build
        behaviour_data.npz/.csv + valid_frames.npz (see build_behaviour_data.py --
        a script port of preprocessing/read_align_timestamps_bonsai.ipynb) and run
        verify_behaviour_data.py's checks on them.
    :param bool force: if False (default), each step that produces a file
        (dFF.npy, the funcimg/behav/merged NWBs, behaviour_data.npz) is skipped
        when its output already exists, so a rerun only redoes what's missing.
        If True, every step redoes its work and overwrites existing output.
        Verification steps always run regardless (they're cheap and are the
        whole point of a rerun).
    """
    rawdata_root = Path(rawdata_root).expanduser().resolve()
    funcimg_preprocessed_root = Path(funcimg_preprocessed_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    subject_dir = rawdata_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject folder does not exist: {subject_dir}")
    full_session_name = _resolve_session_from_tag(subject_dir, session_tag)
    raw_session_path = subject_dir / full_session_name

    funcimg_subject_dir = funcimg_preprocessed_root / subject
    if not funcimg_subject_dir.exists():
        raise FileNotFoundError(f"Subject folder does not exist: {funcimg_subject_dir}")
    funcimg_session_name = _resolve_session_from_tag(funcimg_subject_dir, session_tag)
    funcimg_source_session = funcimg_subject_dir / funcimg_session_name

    output_session = output_root / subject / full_session_name

    if not (raw_session_path / "behav").exists():
        raise FileNotFoundError(f"Missing behav folder: {raw_session_path / 'behav'}")
    if not (funcimg_source_session / "funcimg").exists():
        raise FileNotFoundError(f"Missing funcimg folder: {funcimg_source_session / 'funcimg'}")

    output_session.mkdir(parents=True, exist_ok=True)
    (output_session / "funcimg").mkdir(parents=True, exist_ok=True)
    (output_session / "behav").mkdir(parents=True, exist_ok=True)

    print(f"Raw session:      {raw_session_path}")
    print(f"Funcimg source:   {funcimg_source_session}")
    print(f"Output session:   {output_session}")

    n_steps = 7 if build_behaviour_data_files else 5

    print(f"\n=== Step 1/{n_steps}: ensure dFF.npy exists (compute from suite2p F/Fneu if missing) ===")
    compute_dff.ensure_dff(funcimg_dir=funcimg_source_session / 'funcimg',
                            neuropil_coefficient=dff_neuropil_coefficient, force=force)

    print(f"\n=== Step 2/{n_steps}: funcimg NWB (suite2p + vDAQ TTLs) ===")
    funcimg_nwb_matches = list((output_session / "funcimg").glob("2p_preprocessed_*.nwb"))
    if funcimg_nwb_matches and not force:
        print(f"Already exists, skipping: {funcimg_nwb_matches[0]}")
    else:
        # Aimbot picks the vDAQ .h5 from an unsorted glob('*.tif')[0], so with more than one
        # acquisition in funcimg/ which one's frame TTLs end up in the NWB is filesystem luck.
        raw_tifs = sorted((raw_session_path / 'funcimg').glob('*.tif'))
        if len(raw_tifs) != 1:
            raise RuntimeError(f"Expected exactly one ScanImage .tif in {raw_session_path / 'funcimg'}, found "
                               f"{[p.name for p in raw_tifs]} -- move test acquisitions into a subfolder.")
        convert_2p_to_nwb(
            preprocessed_session_path=str(output_session),
            rawdata_session_path=str(raw_session_path),
            path_tracking=funcimg_source_session,
            subject_id=subject,
            session_id=full_session_name,
            ophys_meta_filename="",
        )

    print(f"\n=== Step 3/{n_steps}: behav NWB (native/unaligned NIDAQ clock) ===")
    behav_nwb_matches = list((output_session / "behav").glob("vrabcd_preprocessed_*.nwb"))
    if behav_nwb_matches and not force:
        print(f"Already exists, skipping: {behav_nwb_matches[0]}")
    else:
        convert_vrabcd_to_nwb(
            preprocessed_session_path=str(output_session),
            rawdata_session_path=str(raw_session_path),
            subject_id=subject,
            session_id=full_session_name,
        )

    print(f"\n=== Step 4/{n_steps}: merged NWB with robust barcode alignment ===")
    merged_nwb_matches = list(output_session.glob("vrabcd_2p_preprocessed_*.nwb"))
    if merged_nwb_matches and not force and not merged_nwb_is_complete(merged_nwb_matches[0]):
        print(f"Existing merged NWB has no NIDAQ TTL tables (an earlier merge crashed) -- rebuilding: "
              f"{merged_nwb_matches[0]}")
        merged_nwb_matches[0].unlink()
        merged_nwb_matches = []
    if merged_nwb_matches and not force:
        print(f"Already exists, skipping: {merged_nwb_matches[0]}")
        merged_nwb_path, fit = merged_nwb_matches[0], None
    else:
        merged_nwb_path, fit = build_merged_nwb_robust(
            preprocessed_funcimg_path=str(output_session),
            preprocessed_behav_path=str(output_session),
            rawdata_path=str(raw_session_path),
        )

    print(f"\n=== Step 5/{n_steps}: verifying the merged NWB ===")
    behav_nwb_path = list((output_session / "behav").glob("vrabcd_preprocessed_*.nwb"))[0]
    funcimg_nwb_path = list((output_session / "funcimg").glob("2p_preprocessed_*.nwb"))[0]
    flags = verify_session.verify(
        merged_nwb_path=merged_nwb_path,
        behav_nwb_path=str(behav_nwb_path),
        funcimg_nwb_path=str(funcimg_nwb_path),
        fit=fit,  # None if step 4 was skipped -- verify_session refits it independently
    )

    if build_behaviour_data_files:
        suite2p_plane_path = funcimg_source_session / 'funcimg' / 'suite2p' / 'plane0'

        print(f"\n=== Step 6/{n_steps}: behaviour_data.npz/.csv + valid_frames.npz "
              "(read_align_timestamps_bonsai.ipynb equivalent) ===")
        behaviour_data_exists = (output_session / 'behaviour_data.npz').exists()
        if behaviour_data_exists and not force:
            print(f"Already exists, skipping: {output_session / 'behaviour_data.npz'}")
            built = None
        else:
            built = build_behaviour_data.build_behaviour_data(
                raw_session_path=raw_session_path,
                output_session_path=output_session,
                suite2p_plane_path=suite2p_plane_path,
            )

        print(f"\n=== Step 7/{n_steps}: verifying behaviour_data.npz ===")
        flags += verify_behaviour_data.verify(
            output_session_path=output_session,
            suite2p_plane_path=suite2p_plane_path,
            built=built,  # None if step 6 was skipped -- verify runs its disk-only checks
        )

    print(f"\nDone. Merged NWB: {merged_nwb_path}")
    return merged_nwb_path, flags


if __name__ == "__main__":
    import defopt
    defopt.run(build_session)
