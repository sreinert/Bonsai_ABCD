"""Pupil diameter from the eye camera of a holography session (behav/Camera2_*.avi).

The pupil glows from the 2p laser light, so it is a saturated, round blob: per frame, threshold a crop
around the eye (>= 220), close the mask, keep the round bright component (convex-hull circularity >= 0.75,
not touching the crop border) nearest the previous pupil position and take its convex-hull area
(equivalent diameter 2 sqrt(area / pi)) and centre. Run from the repo root:

    python -m musicboxes.holography.pupil 001 holo-001 [--camera Camera2] [--max-frames N]

Timing. The video cannot be matched frame by frame to the camera TTLs (behav ai13): sub-001 holo-001 has
104,972 frames in both cameras over the whole session (file written 16:46:42 to 18:28:01) but 304,670
TTL rises at 52.6 Hz, i.e. the cameras saved only ~every third trigger (identical counts in both, so
systematic, e.g. triggers faster than exposure + readout). Frames are therefore put on the vDAQ clock in
two steps: (1) a linear map anchored on the laser glow itself, the first and last frame with a glowing
pupil set to the first and last imaging frame (holo_alignment.npz frame_times); (2) a correction with the
photostimulation trains, which brighten the eye during the 0.475 s spiral train: the lag that maximises
the train-locked brightness per window of 52 trains (half-overlapping, ~4.5 min) is smoothed and
subtracted. In holo-001 the linear map drifted from -0.45 s to +0.5 s over the session (frames were not
dropped uniformly), so step 2 matters. Frames are then cleaned (circularity >= 0.85, diameter within 5
MADs of its ~1 s running median) and the diameter is resampled to the imaging frames.

Outputs (<output>/pupil/): pupil_<camera>_raw.npz (tracking only) and pupil_<camera>.npz (per video frame:
diameter_px, area_px, cx, cy, circ, crop_mean, n_bright, ok, valid, time (corrected) and time_linear on
the vDAQ clock; window lags before / after the correction; ROI and threshold; per imaging frame:
diameter_frames) and pupil_<camera>_qc.png in the analysis folder. --no-retrack redoes only the timing
and cleaning from the raw file.
"""

import argparse
import json
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ceph_paths import ceph_path
import musicboxes.holography.holo_io as hio
import musicboxes.holography.holo_functions as holo


def video_path(raw_behav, camera='Camera2'):
    hits = sorted(p for p in raw_behav.glob(f'{camera}_*.avi') if not p.name.startswith('._'))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one {camera}_*.avi in {raw_behav}, found {[h.name for h in hits]}')
    return hits[0]


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,nb_frames', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)['streams'][0]
    return int(s['width']), int(s['height']), int(s['nb_frames'])


def grab(path, t_s, width, height):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t_s:.3f}', '-i', str(path), '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width)


def round_blobs(img, thr, min_area=300, max_area=40000, min_circ=0.75, close_px=5, skip_border=True):
    """Bright round components as (area, cx, cy, circularity), measured on the convex hull: the pupil
    edge is notched by the dark iris crenations, which inflates the raw perimeter. The thresholded mask
    is closed with a close_px ellipse first; components touching the image border (glare in the crop
    corner) are skipped when skip_border. circularity = 4 pi A_hull / P_hull^2."""
    mask = (img >= thr).astype(np.uint8)
    if close_px:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    H, W = img.shape
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if skip_border and (x == 0 or y == 0 or x + w >= W or y + h >= H):
            continue
        hull = cv2.convexHull(c)
        a = cv2.contourArea(hull)
        if not min_area <= a <= max_area:
            continue
        p = cv2.arcLength(hull, True)
        circ = 4 * np.pi * a / p ** 2 if p > 0 else 0
        if circ < min_circ:
            continue
        m = cv2.moments(hull)
        out.append((a, m['m10'] / m['m00'], m['m01'] / m['m00'], circ))
    return out


def find_roi(path, width, height, duration_s, thr=220, n_samples=40, margin=3.5):
    """Median position / radius of the round saturated blob over frames sampled across the video."""
    hits = []
    for t in np.linspace(0.05, 0.95, n_samples) * duration_s:
        blobs = round_blobs(grab(path, t, width, height), thr, skip_border=False)
        if blobs:
            hits.append(max(blobs, key=lambda b: b[0] * b[3]))
    if not hits:
        raise RuntimeError('no round saturated blob in the sampled frames; is the laser glow visible?')
    h = np.array(hits)
    cx, cy, r = np.median(h[:, 1]), np.median(h[:, 2]), np.sqrt(np.percentile(h[:, 0], 95) / np.pi)
    half = int(max(margin * r, 80))
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, width)), int(min(cy + half, height))
    return {'x0': x0 - x0 % 2, 'y0': y0 - y0 % 2, 'w': (x1 - x0) // 2 * 2, 'h': (y1 - y0) // 2 * 2,
            'cx': cx, 'cy': cy, 'r': r, 'n_sampled_hits': len(hits)}


def track(path, roi, thr=220, max_frames=None, max_jump_px=40, chunk=2000):
    """Per frame: area, diameter, centre of the pupil blob (NaN when none), crop mean brightness and
    number of saturated pixels."""
    w, h = roi['w'], roi['h']
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f"crop={w}:{h}:{roi['x0']}:{roi['y0']}",
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    if max_frames:
        cmd[5:5] = ['-frames:v', str(max_frames)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * chunk)
    res = {k: [] for k in ('area_px', 'cx', 'cy', 'circ', 'crop_mean', 'n_bright')}
    prev = np.array([roi['cx'] - roi['x0'], roi['cy'] - roi['y0']])
    n = 0
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        img = np.frombuffer(buf, np.uint8).reshape(h, w)
        blobs = round_blobs(img, thr)
        near = [b for b in blobs if np.hypot(b[1] - prev[0], b[2] - prev[1]) < max_jump_px + np.sqrt(b[0] / np.pi)]
        if near:
            a, x, y, c = max(near, key=lambda b: b[0])
            prev = np.array([x, y])
        else:
            a = x = y = c = np.nan
        res['area_px'].append(a)
        res['cx'].append(x + roi['x0'])
        res['cy'].append(y + roi['y0'])
        res['circ'].append(c)
        res['crop_mean'].append(img.mean())
        res['n_bright'].append(int((img >= thr).sum()))
        n += 1
        if n % 10000 == 0:
            print(f'  {n} frames', flush=True)
    proc.wait()
    out = {k: np.asarray(v, float) for k, v in res.items()}
    out['diameter_px'] = 2 * np.sqrt(out['area_px'] / np.pi)
    out['ok'] = np.isfinite(out['area_px'])
    return out


def glow_anchors(ok, min_run=200):
    """First and last frame of the glowing period: start / end of the first / last run of >= min_run
    consecutive detected frames (ignores isolated detections before / after imaging)."""
    x = np.r_[0, ok.astype(int), 0]
    starts, ends = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1) - 1
    long = (ends - starts + 1) >= min_run
    return int(starts[long][0]), int(ends[long][-1])


def align(res, alignment):
    ft = alignment['frame_times']
    k0, k1 = glow_anchors(res['ok'])
    slope = (ft[-1] - ft[0]) / (k1 - k0)
    t = ft[0] + (np.arange(len(res['ok'])) - k0) * slope
    return t, {'k_on': k0, 'k_off': k1, 'frame_period_s': slope, 'rate_hz': 1 / slope}


def stim_check(t, crop_mean, onsets, window=(-1, 2)):
    """Mean eye-crop brightness around the photostim train onsets under the timing t."""
    dt = np.median(np.diff(t))
    lags = np.arange(int(window[0] / dt), int(window[1] / dt) + 1)
    idx = np.searchsorted(t, onsets)[:, None] + lags[None]
    ok = (idx >= 0) & (idx < len(t))
    m = np.where(ok, crop_mean[np.clip(idx, 0, len(t) - 1)], np.nan)
    m = m - np.nanmean(m[:, lags < 0], 1, keepdims=True)
    return lags * dt, np.nanmean(m, 0)


def _highpass(x, t, win_s=5.0):
    from scipy.ndimage import uniform_filter1d
    return x - uniform_filter1d(x, max(int(win_s / np.median(np.diff(t))), 3))


def train_lag(t, bright_hp, onsets, lags=np.arange(-1.5, 1.5001, 0.02), train_s=0.475):
    """Lag L (s) maximising the mean high-passed eye brightness during the trains when they are placed at
    onset + L on the video's time axis: the video time is late by L. Returns (L, contrast)."""
    samp = np.linspace(0, train_s, 9)
    vals = []
    for L in lags:
        idx = np.searchsorted(t, (onsets[:, None] + L + samp[None]).ravel())
        idx = idx[(idx > 0) & (idx < len(t))]
        vals.append(bright_hp[idx].mean())
    vals = np.array(vals)
    return lags[np.argmax(vals)], vals.max() - np.median(vals)


def window_lags(t, crop_mean, onsets, win=52, step=26):
    bh = _highpass(crop_mean, t)
    rows = []
    for i in range(0, len(onsets) - win + 1, step):
        L, c = train_lag(t, bh, onsets[i:i + win])
        rows.append((np.median(onsets[i:i + win]), L, c))
    return np.array(rows)                                     # centre time, lag, contrast


def refine_timing(t, crop_mean, onsets, win=52, step=26):
    """Correct the linear frame -> time map with the photostim trains, which brighten the eye: lag per
    window of `win` trains (half-overlapping), running median over 3 windows, interpolated in time
    (constant beyond the first / last window) and subtracted. Returns (t_corrected, before, after)
    with before / after the window lags (centre time, lag, contrast) under each timing."""
    from scipy.ndimage import median_filter
    before = window_lags(t, crop_mean, onsets, win, step)
    lag_s = median_filter(before[:, 1], 3, mode='nearest')
    t_corr = t - np.interp(t, before[:, 0], lag_s)
    after = window_lags(t_corr, crop_mean, onsets, win, step)
    return t_corr, before, after


def clean(res, min_circ=0.85, mad_k=5, win_frames=17):
    """Valid pupil frames: detected, circularity >= min_circ and diameter within mad_k MADs of its ~1 s
    running median (drops blinks and partly occluded frames)."""
    from scipy.ndimage import median_filter
    d = res['diameter_px'].copy()
    ok = res['ok'] & (np.nan_to_num(res['circ']) >= min_circ)
    fill = np.where(ok, d, np.nanmedian(d[ok]))
    med = median_filter(fill, win_frames, mode='nearest')
    dev = np.abs(d - med)
    mad = np.nanmedian(dev[ok])
    return ok & (dev <= mad_k * 1.4826 * mad)


def plot_qc(res, t, ok, info, alignment, frame_img, roi, stim_lin, stim, before, after, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax = axes[0, 0]
    if frame_img is not None:
        ax.imshow(frame_img, cmap='gray')
    ax.add_patch(plt.Rectangle((roi['x0'], roi['y0']), roi['w'], roi['h'], fill=False, color='w', lw=1))
    ax.plot(res['cx'][ok][::50], res['cy'][ok][::50], '.', ms=1, color=holo.ROLE_COLOURS['matched'])
    ax.set_title('Camera frame, crop (box), pupil centres (dots)', fontsize=9)
    ax.set_axis_off()
    ax = axes[0, 1]
    ax.plot(t[ok] / 60, res['diameter_px'][ok], lw=0.2, color=holo.ROLE_COLOURS['projected'])
    ft = alignment['frame_times']
    ax.axvspan(ft[0] / 60, ft[-1] / 60, color=holo.ROLE_COLOURS['other'], alpha=0.3, lw=0, label='imaging')
    ax.set_xlabel('time (min, vDAQ clock, corrected)')
    ax.set_ylabel('pupil diameter (px)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{100 * res['ok'].mean():.1f}% frames detected, {100 * ok.mean():.1f}% valid after cleaning; "
                 f"{info['rate_hz']:.2f} Hz", fontsize=9)
    ax = axes[0, 2]
    ax.plot(before[:, 0] / 60, before[:, 1], 'o-', ms=3, color=holo.ROLE_COLOURS['extra'], label='linear map (glow anchors)')
    ax.plot(after[:, 0] / 60, after[:, 1], 'o-', ms=3, color=holo.ROLE_COLOURS['matched'], label='after correction')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('time (min)')
    ax.set_ylabel('lag of eye brightening vs trains (s)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('timing per window of 52 trains', fontsize=9)
    ax = axes[1, 0]
    ax.hist(res['diameter_px'][ok], bins=80, color=holo.ROLE_COLOURS['projected'])
    ax.set_xlabel('pupil diameter (px), valid frames')
    ax.set_ylabel('frames')
    ax = axes[1, 1]
    ax.hist(res['circ'][res['ok']], bins=60, color='0.5')
    ax.axvline(0.85, color='k', ls='--', lw=0.8)
    ax.set_xlabel('blob circularity (dashed: cleaning cut)')
    ax = axes[1, 2]
    ax.plot(*stim_lin, color=holo.ROLE_COLOURS['extra'], label='linear map')
    ax.plot(*stim, color=holo.ROLE_COLOURS['matched'], label='corrected')
    ax.axvline(0, color='k', lw=0.8)
    ax.axvspan(0, 0.475, color=holo.ROLE_COLOURS['other'], alpha=0.4, lw=0)
    ax.set_xlabel('time from photostim train onset (s)')
    ax.set_ylabel('eye-crop brightness change')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('eye brightness around the 1040 trains', fontsize=9)
    for a_ in axes.flat[1:]:
        a_.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig


RAW_KEYS = ('area_px', 'cx', 'cy', 'circ', 'crop_mean', 'n_bright', 'diameter_px', 'ok')


def postprocess(paths, camera='Camera2', frame_img=None):
    """Timing (glow anchors, then train-based correction), cleaning and resampling to imaging frames,
    from pupil_<camera>_raw.npz (written by run)."""
    out_dir = paths['output'] / 'pupil'
    raw = np.load(out_dir / f'pupil_{camera}_raw.npz')
    res = {k: raw[k] for k in RAW_KEYS}
    roi = {k[4:]: raw[k].item() for k in raw.files if k.startswith('roi_')}
    a = np.load(paths['output'] / 'holo_alignment.npz')
    on = a['trial_onset_time']
    t_lin, info = align(res, a)
    t, before, after = refine_timing(t_lin, res['crop_mean'], on)
    ok = clean(res)
    ft = a['frame_times']
    use = ok & (t >= ft[0] - 1) & (t <= ft[-1] + 1)
    diam_frames = np.interp(ft, t[use], res['diameter_px'][use])
    gap = np.diff(t[use]).max() if use.sum() > 1 else np.nan
    np.savez_compressed(out_dir / f'pupil_{camera}.npz', time=t, time_linear=t_lin, valid=ok,
                        diameter_frames=diam_frames.astype(np.float32), lag_before=before, lag_after=after,
                        threshold=raw['threshold'], **res, **{f'roi_{k}': v for k, v in roi.items()},
                        **{f'align_{k}': v for k, v in info.items()})
    fig = plot_qc(res, t, ok, info, a, frame_img, roi, stim_check(t_lin, res['crop_mean'], on),
                  stim_check(t, res['crop_mean'], on), before, after, paths['analysis'] / f'pupil_{camera}_qc.png')
    plt.close(fig)
    print(f"  pupil detected in {100 * res['ok'].mean():.1f}% of frames, valid {100 * ok.mean():.1f}%; diameter median "
          f"{np.nanmedian(res['diameter_px'][ok]):.1f} px (IQR {np.nanpercentile(res['diameter_px'][ok], 25):.1f}-"
          f"{np.nanpercentile(res['diameter_px'][ok], 75):.1f}); longest gap {gap:.2f} s")
    print(f"  timing: glow frames {info['k_on']}-{info['k_off']} -> {info['rate_hz']:.3f} Hz; train lag per window "
          f"{before[:, 1].min():+.2f}..{before[:, 1].max():+.2f} s before, {after[:, 1].min():+.2f}..{after[:, 1].max():+.2f} s "
          f"after correction (median contrast {np.median(after[:, 2]):.2f})")
    print(f'Saved {out_dir}/pupil_{camera}.npz and pupil_{camera}_qc.png')
    return res, t, ok


def run(mouse, holo_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', camera='Camera2', thr=220,
        max_frames=None, retrack=True):
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    path = video_path(paths['raw_behav'], camera)
    width, height, n_frames = probe(path)
    duration = n_frames / 60.0                        # nominal header rate, only used to pick sample frames
    out_dir = paths['output'] / 'pupil'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f'pupil_{camera}_raw.npz'
    if retrack or not raw_path.exists():
        print(f'{path.name}: {width}x{height}, {n_frames} frames')
        roi = find_roi(path, width, height, duration, thr)
        print(f"  ROI x {roi['x0']}-{roi['x0'] + roi['w']}, y {roi['y0']}-{roi['y0'] + roi['h']}, pupil radius "
              f"~{roi['r']:.0f} px ({roi['n_sampled_hits']}/40 sampled frames with a blob)")
        res = track(path, roi, thr, max_frames)
        np.savez_compressed(raw_path, threshold=thr, **res, **{f'roi_{k}': v for k, v in roi.items()})
    return postprocess(paths, camera, grab(path, duration / 2, width, height))


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--camera', default='Camera2')
    p.add_argument('--threshold', type=int, default=220)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--no-retrack', action='store_true', help='reuse pupil_<camera>_raw.npz, redo timing/cleaning only')
    a = p.parse_args()
    run(a.mouse, a.holo_tag, ceph_path(a.project_root), a.holo_suffix, a.camera, a.threshold, a.max_frames,
        retrack=not a.no_retrack)


if __name__ == '__main__':
    _cli()
