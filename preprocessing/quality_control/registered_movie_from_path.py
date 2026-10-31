"""registered_movie.py, but pointed at a suite2p folder directly instead of root + mouse + session tag.

registered_movie.py finds the session with find_session(root, mouse, tag), which assumes the
sub-<mouse>/ses-... layout and a unique tag. This variant takes the path to the folder that
holds data.bin and the suite2p metadata (db.npy + settings.npy or ops.npy), usually .../plane0,
and uses it as is: no session lookup, no suite2p subpath appended, no projects-root prefix.
A path to data.bin / data_chan2.bin itself is also accepted (its folder is used). Share
prefixes (/Volumes, /ceph, Y:/, Z:/) still go through ceph_path so the same command works on
the Mac and the HPC.

Fetching, caching and rendering are imported unchanged from registered_movie.py (see its
docstring for --mode, --bin-on-fetch, --fetch-only, --from-cache etc.), so the two scripts
cannot drift apart. Only the CLI differs.

The movie label and cache/output names need a mouse and session name. They are guessed from
the path: the session is the folder above suite2p/ (skipping funcimg/), the mouse is the one
above that; without a suite2p/ folder in the path the two folders above the given one are
used. Override with --mouse / --session.

Usage (from preprocessing/quality_control):
    python registered_movie_from_path.py /Volumes/mrsic_flogel/public/projects/.../funcimg/suite2p/plane0 --start 600 --duration 10
    python registered_movie_from_path.py /ceph/.../plane0 --fetch-only                 # e.g. on the HPC
    python registered_movie_from_path.py ~/some_local_copy/plane0 --mouse 002 --session test --duration 0 --mode skip --speed 100
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import ceph_path

from registered_movie import extract_snippet, frames_per_output, load_ops, render


def names_from_path(plane0):
    """(mouse, session) guessed from a suite2p plane folder path."""
    parts = plane0.parts
    if "suite2p" in parts:
        i = len(parts) - 1 - parts[::-1].index("suite2p")
        above = [p for p in parts[:i] if p != "funcimg"]
    else:
        above = list(parts[:-1])
    session = above[-1] if above else plane0.name
    mouse = above[-2] if len(above) > 1 else "unknown"
    return mouse, session


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plane0_dir", nargs="?", help="folder with data.bin + db.npy/ops.npy (or the data.bin itself)")
    p.add_argument("--mouse", help="label for movie/cache names (default: guessed from path)")
    p.add_argument("--session", help="label for movie/cache names (default: guessed from path)")
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
    a = p.parse_args()

    render_downsample = a.downsample
    if a.from_cache:
        npy = a.from_cache
    else:
        if not a.plane0_dir:
            p.error("give plane0_dir (or --from-cache)")
        plane0 = Path(ceph_path(a.plane0_dir)).expanduser()
        if plane0.is_file():
            plane0 = plane0.parent
        if not plane0.is_dir():
            sys.exit(f"Folder not found: {plane0}")
        guessed_mouse, guessed_session = names_from_path(plane0.resolve())
        mouse, session = a.mouse or guessed_mouse, a.session or guessed_session
        fs = load_ops(plane0)["fs"]
        k = frames_per_output(fs, a.speed, a.fps)
        step = k if a.mode == "skip" else 1
        bin_k = k if (a.mode == "mean" and a.bin_on_fetch) else 1
        fetch_ds = a.downsample if a.bin_on_fetch else 1
        npy = extract_snippet(plane0, mouse, session, a.chan, a.start, a.duration,
                              step=step, bin_k=bin_k, downsample=fetch_ds, cache_dir=a.cache_dir)
        if fetch_ds > 1:
            render_downsample = 1
    if a.fetch_only:
        print(f"[done] cached at {npy} (+ .json); copy both and render with --from-cache")
        return
    render(npy, a.out, a.speed, a.fps, a.mode, a.crop, render_downsample, tuple(a.clip), a.gamma, a.crf)


if __name__ == "__main__":
    main()
