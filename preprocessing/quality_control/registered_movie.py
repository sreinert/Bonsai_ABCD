"""Turn a time window of a session's registered imaging stack into a short, sped-up mp4.

Written for sanity checking (motion, bleaching, z-drift, dead/saturated periods) and for
showing data to others. Full sessions are tens of GB, so the script works on time windows
(default 60 s from 600 s in) and only the frames of that window are ever read.

The registered stack is suite2p's plane0/data.bin (data_chan2.bin for channel 2): a flat,
headerless int16 array of shape (nframes, Ly, Lx), with Ly, Lx, nframes and fs in ops.npy.
A time window is therefore one contiguous byte range, read with a single seek -- no suite2p
import needed.

Fetching and rendering are separate steps, because the share can be very slow:
  1. fetch (slow, network): the window's frames are copied to a local cache
     (<cache_dir>/<name>.npy + <name>.json sidecar). Reads go in chunks with progress output;
     the cache is written as .part and only renamed when complete, so a dropped connection
     never leaves a half-filled cache that looks valid. Same parameters -> cache reused.
  2. render (fast, local): speed-up, crop, downsample, contrast, timestamp, mp4 encoding.
     Re-running with other --speed/--clip/--crop etc. only reads the cache.
  --synthetic writes a fake stack (Gaussian cells with calcium-like transients, shot noise,
  slight jitter) into the cache and runs the same render path, to debug without the share.
  --fetch-only / --from-cache split the steps across machines: fetch on the HPC where /ceph
  is local, copy the cached .npy + .json down, render on the Mac.

Speed-up: each output frame stands for k = round(fs * speed / fps) imaging frames, and the
output frame rate is then set to fs * speed / k so the speed-up is exact.
  --mode mean (default): average blocks of k frames. Also denoises the shot-noisy 2p frames.
  --mode skip: take every k-th frame. Only those frames are fetched (one seek each), so far
               fewer bytes cross the network; good for whole-session overviews.
  --bin-on-fetch: with mode mean, average (and --downsample) before caching. The cache is k
               times smaller but is tied to that speed-up.

Contrast is one global percentile clip (--clip, from frames sampled across the window) so
brightness does not flicker between frames. Encoding pipes raw frames to ffmpeg (libx264,
yuv420p: plays in browsers, Keynote, Slack); without ffmpeg it falls back to cv2's mp4v.
A mean projection PNG of the window is saved next to the movie.

Usage (from preprocessing/quality_control; preprocessed_root absolute or relative to the
projects root):
    python registered_movie.py --synthetic --duration 40 --speed 10
    python registered_movie.py SaReMaRa_20260629_GoalProgress/preprocessed 002 t2 --start 600 --duration 10
    python registered_movie.py SaReMaRa_20260629_GoalProgress/preprocessed 002 t2 --duration 0 --mode skip --speed 100
    python registered_movie.py <root> 002 t2 --fetch-only          # e.g. on the HPC
    python registered_movie.py --from-cache ~/qc_movie_cache/<name>.npy --speed 20 --clip 0.5 99.8
For cohort2 (suite2p inside the raw tree) pass the raw root; it is only read.
"""
import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import PROJECTS_ROOT, ceph_path

DEFAULT_SUITE2P_SUBPATH = os.path.join("funcimg", "suite2p", "plane0")
BYTES_PER_PIXEL = 2  # suite2p binaries are int16


# ---------------------------------------------------------------- locating the data

def find_session(basepath, mouse, tag):
    """Return the one session folder under sub-<mouse> (or <mouse>, cohort2) whose name contains tag as a token."""
    print(f"[share] finding session '{tag}' for {mouse} in {basepath} ...", flush=True)
    mouse_dir = f"sub-{mouse}" if not str(mouse).startswith("sub-") else str(mouse)
    mouse_path = Path(basepath) / mouse_dir
    if not mouse_path.is_dir() and (Path(basepath) / str(mouse)).is_dir():
        mouse_path = Path(basepath) / str(mouse)  # cohort2: TAA00000xx folders have no sub- prefix
    if not mouse_path.is_dir():
        sys.exit(f"Mouse folder not found: {mouse_path}")
    pattern = re.compile(rf"(^|[-_]){re.escape(tag)}($|[-_])")
    matches = sorted(f for f in mouse_path.iterdir() if f.is_dir() and pattern.search(f.name))
    if not matches:
        sys.exit(f"No session matching '{tag}' in {mouse_path}")
    if len(matches) > 1:
        sys.exit(f"Several sessions match '{tag}', give a more specific tag: {[m.name for m in matches]}")
    return matches[0]


@functools.lru_cache(maxsize=None)
def _load_ops_cached(plane0_dir):
    plane0_dir = Path(plane0_dir)
    db_path, settings_path = plane0_dir / "db.npy", plane0_dir / "settings.npy"
    print(f"[share] checking suite2p metadata in {plane0_dir} ...", flush=True)
    if db_path.exists():
        # suite2p 1.x: db.npy (~4 MB) + settings.npy (~2 KB) hold the same values as the
        # much larger ops.npy (64 MB in the holography sessions), which matters on a slow share
        print("[share] reading db.npy + settings.npy (suite2p 1.x) ...", flush=True)
        ops = np.load(db_path, allow_pickle=True).item()
        if settings_path.exists():
            ops.setdefault("fs", np.load(settings_path, allow_pickle=True).item().get("fs"))
    else:
        print("[share] reading ops.npy (suite2p 0.14) ...", flush=True)
        ops = np.load(plane0_dir / "ops.npy", allow_pickle=True).item()
    return {"Ly": int(ops["Ly"]), "Lx": int(ops["Lx"]), "nframes": int(ops["nframes"]),
            "fs": float(ops.get("fs") or 45.0)}


def load_ops(plane0_dir):
    """Ly, Lx, nframes, fs from suite2p's db.npy + settings.npy (1.x) or ops.npy (0.14); read once per run."""
    return dict(_load_ops_cached(str(plane0_dir)))


def frame_range(start_s, duration_s, fs, nframes):
    """[f0, f1) imaging frames for a window in seconds; duration 0 means to the end."""
    f0 = int(round(start_s * fs))
    if f0 >= nframes:
        sys.exit(f"--start {start_s}s is beyond the session ({nframes / fs:.0f}s)")
    f1 = nframes if duration_s <= 0 else min(nframes, f0 + int(round(duration_s * fs)))
    return f0, f1


def frames_per_output(fs, speed, fps):
    return max(1, int(round(fs * speed / fps)))


# ---------------------------------------------------------------- fetching (network) and cache

def _downsample(frames, factor):
    """Spatial block mean over factor x factor pixels; frames is (n, Ly, Lx)."""
    if factor <= 1:
        return frames
    n, ly, lx = frames.shape
    ly, lx = ly // factor * factor, lx // factor * factor
    f = frames[:, :ly, :lx].reshape(n, ly // factor, factor, lx // factor, factor)
    return f.mean(axis=(2, 4))


def _cache_name(mouse, session, chan, f0, f1, step, bin_k, downsample):
    name = f"{mouse}_{session}_chan{chan}_f{f0}-{f1}"
    if step > 1:
        name += f"_skip{step}"
    if bin_k > 1 or downsample > 1:
        name += f"_bin{bin_k}_ds{downsample}"
    return name


def _write_cache(cache_dir, name, shape, dtype, meta, fill):
    """Stream frames into <name>.npy via fill(out_memmap); rename from .part only when complete."""
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, part = cache_dir / f"{name}.npy", cache_dir / f"{name}.npy.part"
    out = np.lib.format.open_memmap(part, mode="w+", dtype=dtype, shape=shape)
    try:
        fill(out)
        out.flush()
        del out
    except BaseException:
        del out
        part.unlink(missing_ok=True)
        raise
    os.replace(part, npy)
    (cache_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return npy


def extract_snippet(plane0_dir, mouse, session, chan=1, start_s=600, duration_s=60,
                    step=1, bin_k=1, downsample=1, cache_dir="~/qc_movie_cache", chunk=500):
    """Copy frames [start, start+duration) of the registered binary into the local cache.

    step > 1 reads only every step-th frame (one seek each). bin_k > 1 averages blocks of
    bin_k frames, and downsample block-averages pixels, before caching (cache stored as float32).
    Returns the cache .npy path; an existing cache with the same parameters is reused.
    """
    plane0_dir = Path(plane0_dir)
    ops = load_ops(plane0_dir)
    Ly, Lx, fs = ops["Ly"], ops["Lx"], ops["fs"]
    f0, f1 = frame_range(start_s, duration_s, fs, ops["nframes"])
    name = _cache_name(mouse, session, chan, f0, f1, step, bin_k, downsample)
    npy = Path(cache_dir).expanduser() / f"{name}.npy"
    if npy.exists() and npy.with_suffix(".json").exists():
        print(f"[cache] {npy}")
        return npy

    bin_path = plane0_dir / ("data.bin" if chan == 1 else "data_chan2.bin")
    expected = ops["nframes"] * Ly * Lx * BYTES_PER_PIXEL
    print(f"[share] checking size of {bin_path.name} ...", flush=True)
    actual = os.path.getsize(bin_path)
    if actual != expected:
        print(f"[warn] {bin_path.name} is {actual} bytes, ops implies {expected} "
              f"(nframes={ops['nframes']}, Ly={Ly}, Lx={Lx}); wrong Ly/Lx or truncated file?")

    frame_bytes = Ly * Lx * BYTES_PER_PIXEL
    src_idx = np.arange(f0, f1, step)
    n_out = len(src_idx) // bin_k
    src_idx = src_idx[:n_out * bin_k]
    ly_o, lx_o = (Ly // downsample, Lx // downsample) if downsample > 1 else (Ly, Lx)
    reduce = bin_k > 1 or downsample > 1
    dtype = np.float32 if reduce else np.int16
    meta = {"source": str(bin_path), "mouse": mouse, "session": session, "chan": chan,
            "fs": fs, "f0": f0, "f1": f1, "step": step, "bin_k": bin_k, "downsample": downsample,
            "frames_per_cached": step * bin_k, "Ly": ly_o, "Lx": lx_o, "synthetic": False}
    print(f"[fetch] {bin_path}\n        frames {f0}-{f1} ({(f1 - f0) / fs:.0f}s), "
          f"{len(src_idx)} frames, {len(src_idx) * frame_bytes / 1e9:.2f} GB to read")

    def fill(out):
        t0 = time.time()
        chunk_k = max(bin_k, chunk // bin_k * bin_k)  # read whole bins per chunk
        with open(bin_path, "rb") as f:
            for c0 in range(0, len(src_idx), chunk_k):
                idx = src_idx[c0:c0 + chunk_k]
                if step == 1:
                    f.seek(int(idx[0]) * frame_bytes)
                    raw = np.fromfile(f, dtype=np.int16, count=len(idx) * Ly * Lx)
                else:
                    parts = []
                    for i in idx:
                        f.seek(int(i) * frame_bytes)
                        parts.append(np.fromfile(f, dtype=np.int16, count=Ly * Lx))
                    raw = np.concatenate(parts)
                if raw.size != len(idx) * Ly * Lx:
                    raise IOError(f"short read at frame {idx[0]}: file ended or connection dropped")
                block = raw.reshape(len(idx), Ly, Lx)
                if reduce:
                    block = block.reshape(-1, bin_k, Ly, Lx).mean(axis=1, dtype=np.float32)
                    block = _downsample(block, downsample)
                o0 = c0 // bin_k
                out[o0:o0 + len(block)] = block
                done = c0 + len(idx)
                rate = done * frame_bytes / 1e6 / max(time.time() - t0, 1e-6)
                print(f"\r        {done}/{len(src_idx)} frames, {rate:.0f} MB/s", end="", flush=True)
        print()

    return _write_cache(cache_dir, name, (n_out, ly_o, lx_o), dtype, meta, fill)


def make_synthetic_stack(duration_s=40, fs=45.0, Ly=256, Lx=256, n_cells=60, seed=0,
                         cache_dir="~/qc_movie_cache", chunk=200):
    """Write a fake registered stack to the cache and return its path (always regenerated)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration_s * fs))
    yy, xx = np.mgrid[:Ly, :Lx]
    cy, cx = rng.uniform(8, Ly - 8, n_cells), rng.uniform(8, Lx - 8, n_cells)
    sigma = rng.uniform(2.5, 4.0, n_cells)
    footprints = np.exp(-((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
                        / (2 * sigma[:, None, None] ** 2)).reshape(n_cells, -1).astype(np.float32)
    # calcium-like traces: Poisson spikes convolved with an exponential decay (tau ~0.6 s)
    spikes = rng.poisson(rng.uniform(0.05, 0.5, n_cells)[:, None] / fs, (n_cells, T)).astype(np.float32)
    kernel = np.exp(-np.arange(int(3 * fs)) / (0.6 * fs))
    traces = np.stack([np.convolve(s, kernel)[:T] for s in spikes]) * rng.uniform(150, 400, (n_cells, 1))
    background = 200 + 300 * gaussian_filter(rng.random((Ly, Lx)), 20)
    background = (background - background.min()) / np.ptp(background) * 150 + 150
    background += (footprints * 80).sum(0).reshape(Ly, Lx)  # cells are visible at rest
    bleach = np.exp(-np.arange(T) / (T * 4))  # slow bleaching over the window

    meta = {"source": "synthetic", "mouse": "synthetic", "session": f"seed{seed}", "chan": 1,
            "fs": fs, "f0": 0, "f1": T, "step": 1, "bin_k": 1, "downsample": 1,
            "frames_per_cached": 1, "Ly": Ly, "Lx": Lx, "synthetic": True}

    def fill(out):
        for c0 in range(0, T, chunk):
            c1 = min(T, c0 + chunk)
            act = (traces[:, c0:c1].T @ footprints).reshape(c1 - c0, Ly, Lx)
            mov = (background[None] + act) * bleach[c0:c1, None, None]
            mov = rng.poisson(mov).astype(np.float32)
            for i in range(len(mov)):  # small residual rigid jitter after registration
                mov[i] = np.roll(mov[i], tuple(rng.integers(-1, 2, 2)), axis=(0, 1))
            out[c0:c1] = np.clip(mov, -32768, 32767).astype(np.int16)

    name = f"synthetic_seed{seed}_{Ly}x{Lx}_f0-{T}"
    print(f"[synthetic] {T} frames ({duration_s}s at {fs} Hz), {Ly}x{Lx}, {n_cells} cells")
    return _write_cache(cache_dir, name, (T, Ly, Lx), np.int16, meta, fill)


def load_cache(npy):
    npy = Path(npy).expanduser()
    meta = json.loads(npy.with_suffix(".json").read_text())
    return np.load(npy, mmap_mode="r"), meta


# ---------------------------------------------------------------- rendering (local)

def speedup(stack, k, mode="mean", crop=None, downsample=1):
    """Yield output frames (float32), one per k cached frames."""
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    for i in range(len(stack) // k):
        block = np.asarray(stack[i * k:(i + 1) * k, y0:y1, x0:x1], dtype=np.float32)
        frame = block.mean(axis=0) if mode == "mean" else block[0]
        yield _downsample(frame[None], downsample)[0]


def clip_limits(stack, k, mode, crop, downsample, clip=(1, 99.5), n_sample=200, seed=0):
    """Global intensity limits from output frames sampled across the whole window."""
    n_out = len(stack) // k
    idx = np.sort(np.random.default_rng(seed).choice(n_out, min(n_sample, n_out), replace=False))
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    frames = []
    for i in idx:
        block = np.asarray(stack[i * k:(i + 1) * k, y0:y1, x0:x1], dtype=np.float32)
        frames.append(_downsample((block.mean(axis=0) if mode == "mean" else block[0])[None], downsample)[0])
    lo, hi = np.percentile(np.stack(frames), clip)
    return float(lo), float(hi)


def normalise(frame, lo, hi, gamma=1.0):
    x = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        x = x ** gamma
    return (x * 255).astype(np.uint8)


def annotate(img, text_left, text_right):
    """Burn white-on-black-outline text into the bottom corners of a uint8 frame."""
    h, w = img.shape
    scale = max(0.35, w / 900)
    thick = max(1, int(round(scale * 2)))
    for text, align in ((text_left, "left"), (text_right, "right")):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (6, h - 8) if align == "left" else (w - tw - 6, h - 8)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    return img


def write_mp4(frames, out_path, fps, size, crf=23):
    """Encode an iterable of uint8 (h, w) frames; ffmpeg/libx264 if available, else cv2 mp4v."""
    w, h = size
    ffmpeg = shutil.which("ffmpeg")
    n = 0
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
               "-movflags", "+faststart", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for img in frames:
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
                n += 1
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    else:
        print("[warn] ffmpeg not found, falling back to cv2 mp4v (larger files, may not play in browsers)")
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
        for img in frames:
            vw.write(img)
            n += 1
        vw.release()
    return n


def _fmt_time(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def render(npy, out_dir="~/Desktop/qc_movies", speed=10, fps=30, mode="mean", crop=None,
           downsample=1, clip=(1, 99.5), gamma=1.0, crf=23):
    """Render a cached snippet to mp4 + mean projection PNG; returns the mp4 path."""
    stack, meta = load_cache(npy)
    fs_cached = meta["fs"] / meta["frames_per_cached"]
    k = frames_per_output(fs_cached, speed, fps)
    out_fps = fs_cached * speed / k
    if crop:
        crop = [int(c) for c in crop]
    # libx264 + yuv420p need even dimensions: trim the crop by a pixel if needed
    y0, y1, x0, x1 = crop if crop else (0, stack.shape[1], 0, stack.shape[2])
    h, w = (y1 - y0) // downsample, (x1 - x0) // downsample
    y1 -= (h % 2) * downsample
    x1 -= (w % 2) * downsample
    crop = (y0, y1, x0, x1)
    h, w = h - h % 2, w - w % 2

    lo, hi = clip_limits(stack, k, mode, crop, downsample, clip)
    n_out = len(stack) // k
    t_session = lambda i: (meta["f0"] + i * k * meta["frames_per_cached"]) / meta["fs"]
    label = f"{meta['mouse']} {meta['session']} ch{meta['chan']}"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0s, t1s = meta["f0"] / meta["fs"], meta["f1"] / meta["fs"]
    stem = (f"{meta['mouse']}_{meta['session']}_chan{meta['chan']}_{t0s:.0f}-{t1s:.0f}s"
            f"_x{speed:g}_{mode}")
    out_path = out_dir / f"{stem}.mp4"
    print(f"[render] {n_out} output frames: {k} cached frame(s) each ({mode}), "
          f"{out_fps:.2f} fps -> x{speed:g}, {w}x{h}, clip {lo:.0f}-{hi:.0f}")

    total = np.zeros((h, w), np.float64)

    def frames():
        for i, frame in enumerate(speedup(stack, k, mode, crop, downsample)):
            total[:] += frame[:h, :w]
            img = normalise(frame[:h, :w], lo, hi, gamma)
            yield annotate(img, label, f"{_fmt_time(t_session(i))}  x{speed:g}")
            if i % 50 == 0:
                print(f"\r        {i}/{n_out}", end="", flush=True)

    n = write_mp4(frames(), out_path, out_fps, (w, h), crf)
    print(f"\r        {n}/{n_out}")
    mean_img = normalise(total / max(n, 1), *np.percentile(total / max(n, 1), clip))
    cv2.imwrite(str(out_dir / f"{stem}_mean.png"), mean_img)
    print(f"[saved] {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {n / out_fps:.1f}s)")
    return out_path


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preprocessed_root", nargs="?", help="absolute, or relative to the projects root")
    p.add_argument("mouse", nargs="?", help="e.g. 002 or sub-002")
    p.add_argument("session_tag", nargs="?", help="e.g. t2 or full030")
    p.add_argument("--suite2p-subpath", default=DEFAULT_SUITE2P_SUBPATH)
    p.add_argument("--start", type=float, default=600, help="window start, s (default 600)")
    p.add_argument("--duration", type=float, default=60, help="window length, s; 0 = to end (default 60)")
    p.add_argument("--chan", type=int, choices=[1, 2], default=1)
    p.add_argument("--speed", type=float, default=10, help="playback speed-up (default 10)")
    p.add_argument("--fps", type=float, default=30, help="target output frame rate (default 30)")
    p.add_argument("--mode", choices=["mean", "skip"], default="mean")
    p.add_argument("--bin-on-fetch", action="store_true", help="mode mean: average/downsample before caching")
    p.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    p.add_argument("--downsample", type=int, default=1, help="spatial block-mean factor")
    p.add_argument("--clip", type=float, nargs=2, default=(1, 99.5), metavar=("LO", "HI"),
                   help="intensity percentiles mapped to black/white (default 1 99.5)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--crf", type=int, default=23, help="x264 quality, lower = better/larger (default 23)")
    p.add_argument("--cache-dir", default="~/qc_movie_cache")
    p.add_argument("--out", default="~/Desktop/qc_movies")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fetch-only", action="store_true", help="only fill the cache (e.g. on the HPC)")
    g.add_argument("--from-cache", help="render from a cached .npy (with its .json) instead of fetching")
    g.add_argument("--synthetic", action="store_true", help="render a fake stack, no share needed")
    a = p.parse_args()

    render_downsample = a.downsample
    if a.from_cache:
        npy = a.from_cache
    elif a.synthetic:
        npy = make_synthetic_stack(duration_s=a.duration if a.duration > 0 else 40, cache_dir=a.cache_dir)
    else:
        if not all([a.preprocessed_root, a.mouse, a.session_tag]):
            p.error("give preprocessed_root, mouse and session_tag (or --synthetic / --from-cache)")
        root = Path(ceph_path(a.preprocessed_root))
        if not root.is_absolute():
            root = Path(PROJECTS_ROOT) / root
        ses_dir = find_session(root, a.mouse, a.session_tag)
        plane0 = ses_dir / a.suite2p_subpath
        fs = load_ops(plane0)["fs"]
        k = frames_per_output(fs, a.speed, a.fps)
        step = k if a.mode == "skip" else 1
        bin_k = k if (a.mode == "mean" and a.bin_on_fetch) else 1
        fetch_ds = a.downsample if a.bin_on_fetch else 1
        npy = extract_snippet(plane0, ses_dir.parent.name, ses_dir.name, a.chan, a.start, a.duration,
                              step=step, bin_k=bin_k, downsample=fetch_ds, cache_dir=a.cache_dir)
        if fetch_ds > 1:
            render_downsample = 1
    if a.fetch_only:
        print(f"[done] cached at {npy} (+ .json); copy both and render with --from-cache")
        return
    render(npy, a.out, a.speed, a.fps, a.mode, a.crop, render_downsample, tuple(a.clip), a.gamma, a.crf)


if __name__ == "__main__":
    main()
