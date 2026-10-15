"""Build behav + funcimg + merged NWB files (+ behaviour_data.npz) for one cohort2 session.

Cohort2 counterpart of build_session_nwb.py: same steps, same output layout, same file
names, so cohort2 sessions end up structured exactly like the cohort3/GoalProgress ones:

    <output_root>/sub-<mouse>/<session folder>/
        funcimg/2p_preprocessed_sub-<mouse>_<session>.nwb       vDAQ TTLs + suite2p (+ dF/F)
        behav/vrabcd_preprocessed_sub-<mouse>_<session>.nwb     NIDAQ TTLs + VR log, native NIDAQ clock
        vrabcd_2p_preprocessed_sub-<mouse>_<session>.nwb        merged, behaviour aligned to vDAQ
        behaviour_data.npz / behaviour_data.csv / valid_frames.npz

Usage (in the `2p_preprocess_extraction` conda env, like the rest of nwb_barcoding):

    # one session (protocol tag, ses-XXX prefix, or full folder name):
    python -m nwb_barcoding.build_session_nwb_cohort2 TAA0000066 t6

    # every VR+imaging session of a mouse:
    python -c "from nwb_barcoding.build_session_nwb_cohort2 import build_mouse; build_mouse('TAA0000066')"

What differs from cohort3, and why (details in the module each lives in):
  - raw files are found with cohort2_layout.find_session_files() -- behaviour is a bare
    NIDAQ *_ai.bin + VR position_log.csv instead of Bonsai streams, and suite2p output
    sits inside the raw session tree under funcimg/Session/;
  - the VR log runs on its own clock, so it is first mapped onto the NIDAQ clock by
    fitting VR-logged rewards to NIDAQ reward pulses (cohort2_layout.fit_vr_to_nidaq);
  - the funcimg NWB is built by calling Aimbot_barcoding's MFH2pPreprocessingConverter
    directly (its convert_2p_to_nwb wrapper assumes the tif + suite2p live in
    <session>/funcimg/, which isn't true here);
  - the behav/merged NWBs use cohort2_interfaces.Cohort2VRABCDConverter.

Unchanged from cohort3: the barcode decode + slope/offset fit (barcode_alignment.py), the
dF/F computation (compute_dff.py), and the generic verification checks.

Writes: the output tree above, plus dFF.npy at <session>/funcimg/Session/dff/plane0/ when it
doesn't exist yet (next to suite2p/, where the NWB extractor looks for it -- same as
cohort3, where it is written next to suite2p in the preprocessed tree). Nothing existing in
the raw session tree is modified or overwritten.
"""

import shutil
from pathlib import Path

from . import _aimbot_path
_aimbot_path.ensure_on_path()

from neuroconv.utils import load_dict_from_file  # noqa: E402
from preprocess_alignment.nwb.ophys.mfhconverter import MFH2pPreprocessingConverter  # noqa: E402

from . import barcode_alignment as ba  # noqa: E402
from . import compute_dff  # noqa: E402
from . import cohort2_layout as layout  # noqa: E402
from .cohort2_interfaces import Cohort2VRABCDConverter  # noqa: E402
from . import verify_session_cohort2  # noqa: E402
from . import build_behaviour_data_cohort2  # noqa: E402
from . import verify_behaviour_data_cohort2  # noqa: E402

COHORT2_ROOT = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2'
METADATA_DIR = _aimbot_path.AIMBOT_BARCODING_ROOT / 'preprocess_alignment' / 'nwb' / 'metadata'


def build_funcimg_nwb(files, nwbfile_path, subject_id, session_id):
    """Funcimg NWB: vDAQ FrameTTL/SyncTTL + suite2p segmentation/traces/dF/F.

    Same interfaces, metadata and options as Aimbot's convert_2p_to_nwb (TwoPhotonImaging
    skipped, as cohort3 does), just pointed at cohort2's file locations.
    """
    source_data = dict(
        PMPipelineSegmentation=dict(folder_path=str(files['suite2p_dir']), channel_name='chan1',
                                    idx_dset_separated=None),
        vDAQData=dict(file_path=str(files['vdaq_h5']),
                      ttl_stream_ids_to_names_map=dict(FrameTTL='FrameTTL'),
                      nonttl_stream_ids_to_names_map=dict(SyncTTL='SyncTTL')),
    )
    converter = MFH2pPreprocessingConverter(
        source_data=source_data, ophys_metadata=load_dict_from_file(METADATA_DIR / 'ophys_metadata.yml'),
        verbose=False)

    metadata = converter.get_metadata()
    metadata['NWBFile'].update(session_start_time=_start_time(files), session_id=session_id)
    metadata.setdefault('Subject', {}).update(subject_id=subject_id, sex='M', species='Mus musculus')

    tmp_path = _partial(nwbfile_path)
    converter.run_conversion(metadata=metadata, nwbfile_path=str(tmp_path), overwrite=True,
                             conversion_options=dict(PMPipelineSegmentation=dict(mask_type='image')))
    tmp_path.rename(nwbfile_path)


def _behaviour_converter(files, vr_fit):
    n_channels = layout.ai_channel_count(files['ai_bin'], files['ci_bin'])
    ttl_map = {str(col): name for col, name in layout.AI_CHANNELS.items()
               if col < n_channels and name != 'SyncTTL'}
    source_data = dict(
        NIDAQData=dict(ai_file_path=str(files['ai_bin']), n_channels=n_channels,
                       session_start_time=_start_time(files).replace(tzinfo=None).isoformat(),
                       ttl_channels_to_names_map=ttl_map),
        VRLogData=dict(file_path=str(files['position_log']),
                       vr_to_nidaq_slope=vr_fit['slope'], vr_to_nidaq_intercept=vr_fit['intercept']),
    )
    return Cohort2VRABCDConverter(
        source_data=source_data,
        behavior_metadata=load_dict_from_file(METADATA_DIR / 'behav_metadata_vrabcd.yml'),
        verbose=False)


def build_behav_nwb(files, vr_fit, nwbfile_path, subject_id, session_id):
    """Behav NWB on the native (unaligned, nominal 9000 Hz) NIDAQ clock."""
    converter = _behaviour_converter(files, vr_fit)
    metadata = converter.get_metadata()
    metadata['NWBFile'].update(session_id=session_id)
    metadata.setdefault('Subject', {}).update(subject_id=subject_id, sex='M', species='Mus musculus')
    tmp_path = _partial(nwbfile_path)
    converter.run_conversion(metadata=metadata, nwbfile_path=str(tmp_path), overwrite=True)
    tmp_path.rename(nwbfile_path)


def build_merged_nwb(files, vr_fit, funcimg_nwb_path, behav_nwb_path, merged_nwb_path,
                     nidaq_barcode_height=0.5, vdaq_barcode_height=0.5):
    """Copy the funcimg NWB and append the behaviour data, aligned to the vDAQ clock.

    Same strategy and same fit as build_session_nwb.build_merged_nwb_robust. Always a real
    copy (never Aimbot's copy_nwb, which hard-links when it can -- appending to a hard link
    would silently modify the funcimg NWB too).

    Returns the barcode fit (reused by verification).
    """
    print("Decoding barcodes from funcimg (main) and behav (secondary) NWBs...")
    barcode_funcimg = ba.read_syncttl(funcimg_nwb_path, height=vdaq_barcode_height)
    barcode_behav = ba.read_syncttl(behav_nwb_path, height=nidaq_barcode_height)
    fit = ba.fit_alignment(main_decoded=barcode_funcimg, secondary_decoded=barcode_behav,
                           secondary_nominal_rate=layout.NIDAQ_NOMINAL_RATE)
    match = fit['match']
    ppm_off = 1e6 * (fit['corrected_secondary_rate'] / layout.NIDAQ_NOMINAL_RATE - 1)
    print(f"  {len(match['shared_values'])} shared barcodes, mean|residual|={abs(match['residual_ms']).mean():.2f} ms, "
          f"{match['n_outliers_excluded']} outlier(s) excluded")
    print(f"  delta_t = {fit['delta_t']:.6f} s, corrected NIDAQ rate = {fit['corrected_secondary_rate']:.4f} Hz "
          f"({ppm_off:.1f} ppm off nominal)")

    tmp_path = _partial(merged_nwb_path)
    shutil.copyfile(funcimg_nwb_path, tmp_path)
    converter = _behaviour_converter(files, vr_fit)
    converter.set_alignment_slope(fit['slope'])
    converter.run_conversion(nwbfile_path=str(tmp_path), overwrite=False,
                             align_barcode=True, align_barcode_deltat=fit['delta_t'])
    tmp_path.rename(merged_nwb_path)
    return fit


def _partial(nwbfile_path):
    """Temporary name each NWB is written under, renamed to the real name only once complete.

    The network share has dropped mid-write more than once; since every step skips work
    whose output file exists, a half-written file under the real name would be taken as
    finished on the next run. The dot-prefix also keeps it out of the *.nwb globs used
    downstream.
    """
    nwbfile_path = Path(nwbfile_path)
    return nwbfile_path.with_name('.partial_' + nwbfile_path.name)


def _start_time(files):
    from zoneinfo import ZoneInfo
    # Naive local time labelled UTC -- same convention as the cohort3 NWBs, so session
    # times are comparable across cohorts.
    return layout.session_start_time(files['vdaq_h5']).replace(tzinfo=ZoneInfo('UTC'))


def _existing(path, force):
    if Path(path).exists() and not force:
        print(f"Already exists, skipping: {path}")
        return True
    return False


def build_session(mouse: str, session_tag: str, rawdata_root: str = COHORT2_ROOT,
                  output_root: str = '', dff_neuropil_coefficient: float = 0.7,
                  build_behaviour_data_files: bool = True, force: bool = False):
    """Build behav/funcimg/merged NWBs (+ behaviour_data) for one cohort2 session and verify them.

    :param str mouse: mouse folder name under `rawdata_root`, e.g. "TAA0000066"
    :param str session_tag: protocol tag ("t6"), ses-XXX prefix ("ses-012") or full folder name
    :param str rawdata_root: cohort2 project root (holds <mouse>/ses-*/)
    :param str output_root: defaults to <rawdata_root>/BehaviourAligned_robust
    :param float dff_neuropil_coefficient: used only if dFF.npy has to be computed
    :param bool build_behaviour_data_files: also build behaviour_data.npz/.csv + valid_frames.npz
    :param bool force: redo steps whose output already exists (verification always runs)
    """
    rawdata_root = Path(rawdata_root)
    output_root = Path(output_root) if output_root else rawdata_root / 'BehaviourAligned_robust'
    mouse_dir = rawdata_root / mouse
    session_name = layout.resolve_session(mouse_dir, session_tag)
    files = layout.find_session_files(mouse_dir / session_name)

    subject_id = f"sub-{mouse}"
    out = output_root / subject_id / session_name
    (out / 'funcimg').mkdir(parents=True, exist_ok=True)
    (out / 'behav').mkdir(parents=True, exist_ok=True)
    funcimg_nwb = out / 'funcimg' / f"2p_preprocessed_{subject_id}_{session_name}.nwb"
    behav_nwb = out / 'behav' / f"vrabcd_preprocessed_{subject_id}_{session_name}.nwb"
    merged_nwb = out / f"vrabcd_2p_preprocessed_{subject_id}_{session_name}.nwb"

    print(f"Raw session:    {files['session_path']}")
    print(f"vDAQ recording: {files['vdaq_h5'].name}")
    print(f"Output session: {out}")
    n_steps = 7 if build_behaviour_data_files else 5

    print(f"\n=== Step 1/{n_steps}: ensure dFF.npy exists ===")
    compute_dff.ensure_dff(funcimg_dir=files['funcimg_dir'], neuropil_coefficient=dff_neuropil_coefficient,
                           force=force)

    print(f"\n=== Step 2/{n_steps}: funcimg NWB (suite2p + vDAQ TTLs) ===")
    if not _existing(funcimg_nwb, force):
        build_funcimg_nwb(files, funcimg_nwb, subject_id, session_name)

    print(f"\n=== Step 3/{n_steps}: behav NWB (native NIDAQ clock) ===")
    vr_fit = layout.fit_session_vr_clock(files)
    print(f"VR->NIDAQ clock: {vr_fit['n_matched']}/{vr_fit['n_vr']} VR rewards matched to "
          f"{vr_fit['n_nidaq']} NIDAQ reward pulses, slope={vr_fit['slope']:.8f}, "
          f"intercept={vr_fit['intercept']:.4f} s, max|residual|={abs(vr_fit['residual_ms']).max():.1f} ms")
    if not _existing(behav_nwb, force):
        build_behav_nwb(files, vr_fit, behav_nwb, subject_id, session_name)

    print(f"\n=== Step 4/{n_steps}: merged NWB with robust barcode alignment ===")
    fit = None
    if not _existing(merged_nwb, force):
        fit = build_merged_nwb(files, vr_fit, funcimg_nwb, behav_nwb, merged_nwb)

    print(f"\n=== Step 5/{n_steps}: verifying the NWBs ===")
    flags = verify_session_cohort2.verify(merged_nwb_path=merged_nwb, behav_nwb_path=behav_nwb,
                                          funcimg_nwb_path=funcimg_nwb, fit=fit, vr_fit=vr_fit)

    if build_behaviour_data_files:
        suite2p_plane = files['suite2p_dir'] / 'plane0'
        print(f"\n=== Step 6/{n_steps}: behaviour_data.npz/.csv + valid_frames.npz ===")
        built = None
        if not _existing(out / 'behaviour_data.npz', force):
            built = build_behaviour_data_cohort2.build_behaviour_data(
                files=files, output_session_path=out, vr_fit=vr_fit)

        print(f"\n=== Step 7/{n_steps}: verifying behaviour_data.npz ===")
        flags += verify_behaviour_data_cohort2.verify(output_session_path=out,
                                                      suite2p_plane_path=suite2p_plane, built=built)

    layout.load_ai.cache_clear()  # release the ~1 GB analog array before the next session
    print(f"\nDone. Merged NWB: {merged_nwb}")
    return merged_nwb, flags


def build_mouse(mouse: str, rawdata_root: str = COHORT2_ROOT, output_root: str = '',
                sessions: list = None, force: bool = False):
    """Run build_session over every VR+imaging session of a mouse (or `sessions`, if given).

    A failing session is logged with its traceback and the batch moves on; a summary table
    (status + number of verification flags) is printed at the end and returned.
    """
    import traceback
    if sessions is None:
        sessions = layout.list_vr_imaging_sessions(Path(rawdata_root) / mouse)
    print(f"{len(sessions)} session(s) to build: {sessions}")

    results = []
    for session in sessions:
        print(f"\n\n########## {mouse} {session} ##########", flush=True)
        try:
            _, flags = build_session(mouse, session, rawdata_root=rawdata_root,
                                     output_root=output_root, force=force)
            results.append((session, 'OK', len(flags), ''))
        except Exception as e:
            traceback.print_exc()
            layout.load_ai.cache_clear()
            results.append((session, 'FAILED', None, f"{type(e).__name__}: {e}"))
            if not Path(rawdata_root).exists():
                # The network share dropped: every remaining session would fail instantly.
                # Stop, so a rerun (which skips finished steps) picks up from here.
                print(f"\n{rawdata_root} is no longer reachable -- stopping the batch.")
                done = {r[0] for r in results}
                results += [(s, 'NOT RUN', None, 'share unreachable') for s in sessions if s not in done]
                break

    print('\n\n########## SUMMARY ##########')
    for session, status, n_flags, note in results:
        flag_txt = f"{n_flags} flag(s)" if n_flags is not None else ''
        print(f"{session:45s} {status:7s} {flag_txt:10s} {note}")
    return results


if __name__ == "__main__":
    import defopt
    defopt.run(build_session)
