"""
Split a suite2p run over several concatenated sessions back into one suite2p folder per session.

Why: running suite2p once on several sessions (TIFFs copied into one folder) gives one set of ROIs
shared by all sessions, so no ROICaT matching and no cellpose run-to-run variability. Every
time-resolved output (F, Fneu, spks, the chan2 traces, per-frame registration offsets) then covers
all sessions back to back. This script cuts them apart again.

How sessions are identified: suite2p (v1.x) stores the input TIFFs in `file_list` and their
lengths in `frames_per_file` (in db.npy / ops.npy). Files are grouped into sessions:
  - by parent folder, if the TIFFs came from more than one folder (`frames_per_folder` has >1 entry);
  - otherwise by ScanImage acquisition name, i.e. the file name without the trailing `_00001` counter,
    so a session written as several `_0000N.tif` chunks stays together.
Each session is matched to an original session folder (`ses-..._date-YYYYMMDDTHHMMSS...`) in
`--sessions-root` (default: the sub-XXX folder that holds the concatenated session) by the
`YYYYMMDD_HHMMSS` timestamp at the start of the ScanImage file name: same day, nearest time. A
session with no match is named after its acquisition. Note: an extra acquisition in the input
folder (e.g. a ScanImage test run) shows up here as an extra session, so check the printed table.

What is written: `<output_dir>/<session>/funcimg/suite2p/planeN/`, with output_dir one folder per
concatenated run (e.g. `cohort1/concatenated_runs/sub-001/concat007009`), holding
  - F, Fneu, spks, F_chan2, Fneu_chan2 (and any other .npy whose last axis is nframes): sliced in time;
  - ops.npy, reg_outputs.npy, db.npy: per-frame arrays (yoff, xoff, corrXY, badframes, ...) sliced,
    nframes/file_list/frames_per_file/paths updated, plus a `concat_split` entry with the source
    folder and frame range;
  - stat, iscell, redcell, settings, images, ...: copied unchanged (ROIs are shared across sessions).
Not changed: meanImg/meanImgE/refImg/max_proj and the ROI statistics in stat.npy (skew, std, ...)
are still computed over all sessions together. spks were deconvolved on the concatenated trace,
so the first few frames after a session junction carry a little decay from the previous session.
dF/F is not split: compute it per session from the split F/Fneu (nwb_barcoding.compute_dff does
this when dFF.npy is missing), so each session gets its own baseline.

The registered binaries (data.bin, data_chan2.bin; ~85 GB each for two 30 min sessions) are only
split with --copy-bin. Copying them over the SMB mount is slow and can drop, so run that on the HPC.

Usage (from the repo root, in the uv env):
    uv run python preprocessing/suite2p/split_concat_suite2p.py \
        /ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/preprocessed/sub-001/ses-seq3-test_concat007009/funcimg/suite2p \
        /ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/concatenated_runs/sub-001/concat007009
    add --dry-run to only print the session split.
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import ceph_path

DICT_FILES = ('ops.npy', 'reg_outputs.npy', 'db.npy')
BIN_FILES = ('data.bin', 'data_chan2.bin', 'data_raw.bin', 'data_chan2_raw.bin')
ACQ_RE = re.compile(r'^(?P<stem>.*?)(?:_\d{5})?\.tiff?$', re.IGNORECASE)
FILE_TIME_RE = re.compile(r'(\d{8})_(\d{6})')
SES_TIME_RE = re.compile(r'date-(\d{8})T(\d{6})')
MAX_MATCH_GAP_H = 3  # warn if the nearest session folder starts more than this many hours away


def load_dict(path):
    return np.load(path, allow_pickle=True).item()


def find_sessions(db):
    """Group the input files into sessions. Returns a list of dicts with files, frame range and acquisition."""
    files = [Path(f) for f in db['file_list']]
    frames = np.asarray(db['frames_per_file'], dtype=int)
    by_folder = len(np.atleast_1d(db.get('frames_per_folder', [0]))) > 1

    sessions = []
    start = 0
    for f, n in zip(files, frames):
        m = ACQ_RE.match(f.name)
        acq = m.group('stem') if m else f.stem
        key = str(f.parent) if by_folder else (str(f.parent), acq)
        if not sessions or sessions[-1]['key'] != key:
            sessions.append(dict(key=key, acquisition=acq, files=[], frames_per_file=[], start=start, stop=start))
        sessions[-1]['files'].append(str(f))
        sessions[-1]['frames_per_file'].append(int(n))
        sessions[-1]['stop'] += n
        start += n
    for s in sessions:
        del s['key']
    return sessions


def match_session_folder(acquisition, sessions_root, exclude):
    """Original session folder whose date-YYYYMMDDTHHMMSS is on the same day and closest in time."""
    m = FILE_TIME_RE.search(acquisition)
    if m is None or sessions_root is None:
        return None
    t_acq = datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M%S')
    candidates = []
    for d in sessions_root.iterdir():
        ms = SES_TIME_RE.search(d.name)
        if not d.is_dir() or d.name == exclude or ms is None:
            continue
        t_ses = datetime.strptime(''.join(ms.groups()), '%Y%m%d%H%M%S')
        if t_ses.date() == t_acq.date():
            candidates.append((abs((t_ses - t_acq).total_seconds()), d.name))
    if not candidates:
        return None
    gap, name = min(candidates)
    if gap > MAX_MATCH_GAP_H * 3600:
        print(f"  WARNING: {acquisition} matched to {name}, but they start {gap / 3600:.1f} h apart")
    return name


def slice_dict(d, start, stop, nframes_total, extra):
    """Slice per-frame arrays (first axis == nframes_total) in a suite2p ops-like dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nframes_total:
            out[k] = v[start:stop].copy()
        else:
            out[k] = v
    out.update({k: v for k, v in extra.items() if k in d or k == 'concat_split'})
    return out


def split_binary(src, dst, start, stop, Ly, Lx, chunk=2000):
    """Copy frames [start, stop) of a suite2p int16 binary in chunks."""
    frame_bytes = Ly * Lx * 2
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        fi.seek(start * frame_bytes)
        for i in range(start, stop, chunk):
            n = min(chunk, stop - i)
            buf = fi.read(n * frame_bytes)
            if len(buf) != n * frame_bytes:
                raise IOError(f"{src} ended early at frame {i}")
            fo.write(buf)
    print(f"    {src.name}: frames {start}-{stop} -> {dst}")


def split_plane(plane_dir, out_plane, sess, nframes_total, copy_bin):
    out_plane.mkdir(parents=True, exist_ok=True)
    start, stop = sess['start'], sess['stop']
    db = load_dict(plane_dir / 'db.npy') if (plane_dir / 'db.npy').exists() else {}
    Ly, Lx = db.get('Ly'), db.get('Lx')

    split_info = dict(source=str(plane_dir), frame_start=int(start), frame_stop=int(stop),
                      nframes_concat=int(nframes_total), files=sess['files'])
    extra = dict(
        nframes=int(stop - start),
        file_list=sess['files'],
        frames_per_file=np.asarray(sess['frames_per_file']),
        frames_per_folder=np.array([stop - start]),
        first_files=np.array([True] + [False] * (len(sess['files']) - 1)),
        save_path=str(out_plane),
        save_path0=str(out_plane.parents[1]),
        fast_disk=str(out_plane),
        ops_path=str(out_plane / 'ops.npy'),
        settings_path=str(out_plane / 'settings.npy'),
        db_path=str(out_plane / 'db.npy'),
        reg_file=str(out_plane / 'data.bin'),
        reg_file_chan2=str(out_plane / 'data_chan2.bin'),
        raw_file=str(out_plane / 'data_raw.bin'),
        raw_file_chan2=str(out_plane / 'data_chan2_raw.bin'),
        concat_split=split_info,
    )

    for f in sorted(plane_dir.iterdir()):
        if f.is_dir():
            continue
        if f.name in BIN_FILES:
            if copy_bin:
                split_binary(f, out_plane / f.name, start, stop, Ly, Lx)
            continue
        if f.name in DICT_FILES:
            np.save(out_plane / f.name, slice_dict(load_dict(f), start, stop, nframes_total, extra))
            continue
        if f.suffix == '.npy':
            a = np.load(f, allow_pickle=True)
            if a.dtype != object and a.ndim >= 1 and a.shape[-1] == nframes_total:
                np.save(out_plane / f.name, a[..., start:stop])
                continue
        shutil.copy2(f, out_plane / f.name)  # per-ROI files (stat, iscell, ...), settings, images

    with open(out_plane / 'concat_split.json', 'w') as fh:
        json.dump(split_info, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Split a suite2p run over concatenated sessions into per-session folders.")
    parser.add_argument('suite2p_dir', help="suite2p output folder of the concatenated run (the one holding plane0/)")
    parser.add_argument('output_dir', help="folder for this concatenated run; writes <output_dir>/<session>/funcimg/suite2p")
    parser.add_argument('--sessions-root', default=None,
                        help="folder with the original session folders to name the splits after (default: the sub-XXX "
                             "folder containing the concatenated session)")
    parser.add_argument('--copy-bin', action='store_true', help="also split data.bin / data_chan2.bin (large, slow over SMB)")
    parser.add_argument('--force', action='store_true', help="overwrite existing split outputs")
    parser.add_argument('--dry-run', action='store_true', help="only print how the run would be split")
    args = parser.parse_args()

    suite2p_dir = Path(ceph_path(args.suite2p_dir))
    if suite2p_dir.name.startswith('plane'):
        suite2p_dir = suite2p_dir.parent
    output_dir = Path(ceph_path(args.output_dir))
    if 'rawdata' in output_dir.parts:
        sys.exit("Refusing to write into rawdata/.")
    concat_session = suite2p_dir.parents[1]  # .../sub-XXX/<concat session>/funcimg/suite2p
    sessions_root = Path(ceph_path(args.sessions_root)) if args.sessions_root else concat_session.parent

    plane_dirs = sorted(d for d in suite2p_dir.iterdir() if d.is_dir() and re.fullmatch(r'plane\d+', d.name))
    if not plane_dirs:
        sys.exit(f"No planeN folders in {suite2p_dir}")
    db = load_dict(plane_dirs[0] / 'db.npy')
    nframes_total = int(db['nframes'])
    if int(np.sum(db['frames_per_file'])) != nframes_total:
        sys.exit(f"frames_per_file sums to {np.sum(db['frames_per_file'])}, but nframes is {nframes_total}")

    sessions = find_sessions(db)
    for s in sessions:
        s['name'] = match_session_folder(s['acquisition'], sessions_root, concat_session.name) or f"ses-{s['acquisition']}"
    names = [s['name'] for s in sessions]
    if len(set(names)) != len(names):
        sys.exit(f"Several acquisitions map to the same session folder: {names}. Pass --sessions-root or split by hand.")

    print(f"{suite2p_dir}\n  {nframes_total} frames, {len(sessions)} session(s), {len(plane_dirs)} plane(s):")
    for s in sessions:
        print(f"  frames {s['start']:>7}-{s['stop']:<7} ({s['stop'] - s['start']} frames, {len(s['files'])} file(s)) "
              f"{s['acquisition']} -> {s['name']}")
    if len(sessions) < 2:
        print("  Only one session found: nothing to split.")
        return
    if args.dry_run:
        return

    for s in sessions:
        out_s2p = output_dir / s['name'] / 'funcimg' / 'suite2p'
        if out_s2p.exists() and not args.force:
            print(f"  {out_s2p} exists, skipping (use --force to overwrite)")
            continue
        print(f"  writing {out_s2p}")
        for f in suite2p_dir.iterdir():  # top-level db.npy / settings.npy
            if f.is_file():
                out_s2p.mkdir(parents=True, exist_ok=True)
                if f.name == 'db.npy':
                    d = load_dict(f)
                    d.update(file_list=s['files'], first_files=np.array([True] + [False] * (len(s['files']) - 1)),
                             save_path0=str(out_s2p.parent), fast_disk=str(out_s2p.parent))
                    np.save(out_s2p / f.name, d)
                else:
                    shutil.copy2(f, out_s2p / f.name)
        for plane_dir in plane_dirs:
            split_plane(plane_dir, out_s2p / plane_dir.name, s, nframes_total, args.copy_bin)
    print("Done.")


if __name__ == '__main__':
    main()
