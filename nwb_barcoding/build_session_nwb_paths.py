"""Build behav + funcimg + merged NWB files for one session from explicit paths.

Copy of build_session_nwb.build_session() for suite2p output that doesn't sit in a
`<root>/sub-XXX/<session>` tree, e.g. the per-session splits of a suite2p run over
concatenated sessions (preprocessing/suite2p/split_concat_suite2p.py), which live at
`concatenated_runs/sub-001/concat007009/<session>/funcimg/suite2p`. Instead of a
subject + session tag resolved against root folders, it takes:
  - the raw session folder (behav/ with the Bonsai streams, funcimg/ with the ScanImage
    .tif + vDAQ .h5), e.g. `rawdata/sub-001/ses-seq3-007_date-20261107T151431-t9`;
  - the suite2p folder itself, e.g. `.../concat007009/ses-seq3-007_date-20261107T151431-t9/funcimg/suite2p`.
Kept as a separate module so build_session_nwb.py (and the HPC array jobs that run it)
stay untouched; the merge step (build_merged_nwb_robust) is imported from there, not copied.

The suite2p folder must be named `suite2p` and sit inside a `funcimg/` folder: Aimbot's
(unmodified) convert_2p_to_nwb reads `<path_tracking>/funcimg/suite2p` and compute_dff
reads/writes next to it (`<...>/funcimg/dff/plane0/dFF.npy`), so the folder two levels
up is passed on as `path_tracking`. dFF.npy is therefore written next to the given
suite2p folder if it's missing, i.e. per split session, with its own baseline.

Output goes to `<output_root>/<subject>/<raw session folder name>/`, with subject taken
from the raw session's parent folder unless given.

Usage (from the repo root, in the nwb_barcoding env):
    python -m nwb_barcoding.build_session_nwb_paths \\
        /ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/rawdata/sub-001/ses-seq3-007_date-20261107T151431 \\
        /ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/concatenated_runs/sub-001/concat007009/ses-seq3-007_date-20261107T151431-t9/funcimg/suite2p \\
        /ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/BehaviourAligned_concat007009
"""

from pathlib import Path

from ceph_paths import ceph_path
from . import _aimbot_path
_aimbot_path.ensure_on_path()

from preprocess_alignment.nwb_conversion_2p import convert_2p_to_nwb
from preprocess_alignment.nwb_conversion_vrabcd import convert_vrabcd_to_nwb

from .build_session_nwb import build_merged_nwb_robust, merged_nwb_is_complete
from . import verify_session
from . import compute_dff
from . import build_behaviour_data
from . import verify_behaviour_data


def build_session_from_paths(raw_session_path: str, suite2p_path: str, output_root: str, *,
                             subject: str = '', dff_neuropil_coefficient: float = 0.7,
                             build_behaviour_data_files: bool = True, force: bool = False):
    """Build behav/funcimg/merged NWBs for one session from explicit paths and verify the result.

    :param str raw_session_path: raw session folder holding behav/ and funcimg/, e.g.
        ".../cohort1/rawdata/sub-001/ses-seq3-007_date-20261107T151431"
    :param str suite2p_path: the suite2p output folder (holding plane0/), which must be
        `<something>/funcimg/suite2p`
    :param str output_root: where to write the behav/funcimg/merged NWBs; the session goes to
        <output_root>/<subject>/<raw session folder name>. Use a NEW tree, never an existing
        BehaviourAligned output.
    :param str subject: e.g. "sub-001"; default: the raw session's parent folder name
    :param float dff_neuropil_coefficient: neuropil factor if dFF.npy needs computing (see compute_dff.py)
    :param bool build_behaviour_data_files: also build behaviour_data.npz/.csv + valid_frames.npz and verify them
    :param bool force: redo every step and overwrite existing output (default: skip steps whose output exists)
    """
    # Share paths may be given with any machine's prefix (/Volumes, /ceph, Y:/); see ceph_paths.py.
    raw_session_path = Path(ceph_path(raw_session_path)).expanduser().resolve()
    suite2p_path = Path(ceph_path(suite2p_path)).expanduser().resolve()
    output_root = Path(ceph_path(output_root)).expanduser().resolve()
    if suite2p_path.name.startswith('plane'):
        suite2p_path = suite2p_path.parent

    if suite2p_path.name != 'suite2p' or suite2p_path.parent.name != 'funcimg':
        raise ValueError(f"suite2p_path must be <...>/funcimg/suite2p (Aimbot's converter and compute_dff "
                         f"expect that layout), got {suite2p_path}")
    if not (suite2p_path / 'plane0' / 'stat.npy').exists():
        raise FileNotFoundError(f"No suite2p output at {suite2p_path / 'plane0'}")
    if not (raw_session_path / "behav").exists():
        raise FileNotFoundError(f"Missing behav folder: {raw_session_path / 'behav'}")
    funcimg_source_session = suite2p_path.parents[1]  # Aimbot's path_tracking: <this>/funcimg/suite2p

    subject = subject or raw_session_path.parent.name
    full_session_name = raw_session_path.name
    output_session = output_root / subject / full_session_name

    output_session.mkdir(parents=True, exist_ok=True)
    (output_session / "funcimg").mkdir(parents=True, exist_ok=True)
    (output_session / "behav").mkdir(parents=True, exist_ok=True)

    print(f"Raw session:      {raw_session_path}")
    print(f"Suite2p source:   {suite2p_path}")
    print(f"Output session:   {output_session}")

    n_steps = 7 if build_behaviour_data_files else 5

    print(f"\n=== Step 1/{n_steps}: ensure dFF.npy exists (compute from suite2p F/Fneu if missing) ===")
    compute_dff.ensure_dff(funcimg_dir=suite2p_path.parent,
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
        suite2p_plane_path = suite2p_path / 'plane0'

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
    defopt.run(build_session_from_paths)
