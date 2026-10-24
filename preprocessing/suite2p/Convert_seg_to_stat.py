"""
Replace suite2p's ROIs with cellpose masks (a *_seg.npy file saved from the cellpose GUI,
e.g. meanImg_seg.npy or meanImg_chan2_seg.npy) and re-run extraction, deconvolution and
classification on the registered binary.

Works with both suite2p versions:
  - suite2p >= 1.0 (plane0 contains db.npy and settings.npy): the cellpose stat is passed
    to suite2p.run_plane, which skips registration (reg_outputs.npy already exists) and
    detection, and runs extraction (both channels if present), deconvolution and
    classification with the session's saved settings. db.npy and settings.npy are only
    modified in memory, so the HPC paths stored in them stay intact.
  - suite2p 0.x (plane0 contains only ops.npy, e.g. cohort2 on the HPC): the original
    step-by-step extraction using ops.npy.

Writes stat.npy, F.npy, Fneu.npy, spks.npy, iscell.npy and ops.npy into the suite2p plane0
folder, overwriting existing ones (suite2p >= 1.0 also writes F_chan2.npy, Fneu_chan2.npy
and zcorr.npy). Consider backing up this folder first, and do not run it on suite2p output
inside rawdata/.

Usage: python Convert_seg_to_stat.py <basepath> <mouse> <session> [--seg FILE] [--device DEV]
  basepath: project root containing the mouse folders
  mouse:    mouse folder name, with or without the 'sub-' prefix (e.g. '004', 'TAA0000066')
  session:  session folder name, or any unique part of it (e.g. 't17', '2LM002', a date)
  seg:      cellpose file name in plane0 (default: the only *_seg.npy file there)
  device:   torch device for suite2p >= 1.0 ('cpu', 'cuda', 'mps'; default 'cpu')
"""
import argparse
import numpy as np
from pathlib import Path

import suite2p
from suite2p.detection import roi_stats

parser = argparse.ArgumentParser(description="Convert cellpose masks to a suite2p stat file and re-extract traces.")
parser.add_argument("basepath", type=str, help="Project root containing the mouse folders")
parser.add_argument("mouse", type=str, help="Mouse folder name, e.g. '004', 'sub-004' or 'TAA0000066'")
parser.add_argument("session", type=str, help="Session folder name or a unique part of it")
parser.add_argument("--seg", type=str, default=None, help="Cellpose *_seg.npy file name in plane0 (default: the only one there)")
parser.add_argument("--device", type=str, default="cpu", help="Torch device for suite2p >= 1.0 (default: cpu)")
args = parser.parse_args()

# Mouse folder: as given (e.g. cohort2 'TAA0000066'), otherwise with the 'sub-' prefix
mouse_path = Path(args.basepath) / args.mouse
if not mouse_path.is_dir():
    mouse_path = Path(args.basepath) / f"sub-{args.mouse}"

# Session folder: exact name match first, otherwise a unique substring match
session_path = mouse_path / args.session
if not session_path.is_dir():
    matches = [f for f in mouse_path.iterdir() if f.is_dir() and args.session in f.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one session folder matching '{args.session}' in {mouse_path}, found {[m.name for m in matches]}")
    session_path = matches[0]

# suite2p output can sit at funcimg/suite2p or funcimg/Session/suite2p depending on cohort
ops_files = sorted(session_path.glob("**/suite2p/plane0/ops.npy"))
if not ops_files:
    raise FileNotFoundError(f"No suite2p/plane0/ops.npy found under {session_path}")
wd = ops_files[0].parent
session = session_path.name
print(f"Processing session: {session}")
print(f"suite2p folder: {wd}")

# Read in mask data from cellpose
if args.seg is not None:
    seg_path = wd / args.seg
else:
    seg_files = sorted(wd.glob("*_seg.npy"))
    if len(seg_files) != 1:
        raise FileNotFoundError(f"Expected one *_seg.npy file in {wd}, found {[f.name for f in seg_files]}; choose one with --seg")
    seg_path = seg_files[0]
print(f"Cellpose masks: {seg_path.name}")
cellpose_masks = np.load(seg_path, allow_pickle=True).item()
masks = cellpose_masks['masks']


def masks_to_stat(masks, Ly, Lx, **roi_stats_kwargs):
    """Mimic the stat array made by suite2p from a cellpose label image."""
    stat = []
    for u in np.unique(masks)[1:]:
        ypix, xpix = np.nonzero(masks == u)
        npix = len(ypix)
        stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32), 'med': [np.mean(ypix), np.mean(xpix)]})
    stat = np.array(stat)
    # Fills in remaining roi properties to make it compatible with the rest of the suite2p pipeline/GUI
    return roi_stats(stat, Ly, Lx, **roi_stats_kwargs)


def run_suite2p_v1(wd, masks, device):
    """suite2p >= 1.0: hand the cellpose stat to run_plane with the session's db/settings."""
    db = np.load(wd/'db.npy', allow_pickle=True).item()
    settings = np.load(wd/'settings.npy', allow_pickle=True).item()

    # Point the binaries/save paths at this folder (db may hold HPC /ceph paths). Not saved back.
    db['save_path'] = str(wd)
    db['fast_disk'] = str(wd)
    db['save_path0'] = str(wd.parents[1])
    db['data_path'] = [str(wd.parents[1])]
    db['reg_file'] = str(wd/'data.bin')
    if db.get('nchannels', 1) > 1:
        db['reg_file_chan2'] = str(wd/'data_chan2.bin')

    # Detection must be enabled or the pipeline stops before extraction; with a given stat it
    # skips the detection step itself. Registration is skipped because reg_outputs.npy exists.
    settings['run']['do_detection'] = True
    settings['classification']['use_builtin_classifier'] = True
    settings['torch_device'] = device

    stat = masks_to_stat(masks, db['Ly'], db['Lx'], diameter=settings['diameter'])
    print(f"{len(stat)} ROIs from cellpose")
    try:
        suite2p.run_plane(db, settings, stat=stat)
    except UnboundLocalError as e:
        # suite2p 1.x bug: with a given stat, pipeline() skips detection but still returns
        # `detect_outputs`, so it crashes after all outputs (stat, F, Fneu, F_chan2, Fneu_chan2,
        # spks, iscell, zcorr) are saved and before run_plane saves ops.npy. Save ops.npy here.
        if "detect_outputs" not in str(e):
            raise
        print("suite2p crashed after saving outputs (known detect_outputs bug); writing ops.npy")
        reg_outputs = np.load(wd/'reg_outputs.npy', allow_pickle=True).item()
        ops = {**db, **suite2p.default_settings(), **settings, **reg_outputs}
        np.save(wd/'ops.npy', ops)


def run_suite2p_v0(wd, masks):
    """suite2p 0.x: original step-by-step extraction from ops.npy and the registered binary."""
    ops = np.load(wd/'ops.npy', allow_pickle=True).item()
    Lx = ops['Lx']
    Ly = ops['Ly']
    f_reg = suite2p.io.BinaryFile(Ly, Lx, wd/'data.bin')

    stat = masks_to_stat(masks, Ly, Lx)
    print(f"{len(stat)} ROIs from cellpose")

    # Feed these values into the wrapper functions
    stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,ops=ops)

    # Do cell classification
    classfile = suite2p.classification.builtin_classfile
    iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

    # Apply preprocessing step for deconvolution
    dF = F.copy() - ops['neucoeff']*Fneu
    dF = suite2p.extraction.preprocess(
            F=dF,
            baseline=ops['baseline'],
            win_baseline=ops['win_baseline'],
            sig_baseline=ops['sig_baseline'],
            fs=ops['fs'],
            prctile_baseline=ops['prctile_baseline']
        )
    # Identify spikes
    spks = suite2p.extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

    # Overwrite files in wd folder (consider backing up this folder first)
    np.save(wd/'F.npy', F)
    np.save(wd/'Fneu.npy', Fneu)
    np.save(wd/'iscell.npy', iscell)
    np.save(wd/'ops.npy', ops)
    np.save(wd/'spks.npy', spks)
    np.save(wd/'stat.npy', stat)


if (wd/'db.npy').exists() and (wd/'settings.npy').exists():
    run_suite2p_v1(wd, masks, args.device)
else:
    run_suite2p_v0(wd, masks)

print(f"Finished updating session: {session}")
