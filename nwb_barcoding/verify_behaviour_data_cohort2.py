"""Sanity checks for a cohort2 behaviour_data.npz/valid_frames.npz.

Disk-based checks (shapes vs. the merged NWB, Ca_traces == spks[:, valid_frames]) and the
raw-vs-npz event counts are reused from verify_behaviour_data.py unchanged. The metrics
check is cohort2's own, since the raw reference is the VR position log rather than a
Bonsai event dataframe.
"""

from pathlib import Path

import numpy as np

from . import cohort2_layout as layout
from .verify_behaviour_data import (_flag, check_shapes, check_frame_overlap, check_event_counts,
                                    check_ca_traces_identity)


def check_vr_metrics(built, behav_data):
    flags = []
    vr_log = built['vr_log']
    n_vr_rewards = len(layout.vr_reward_times(vr_log))
    npz_rewards = int(np.nansum(behav_data['rewards']))
    n_vr_licks = int((vr_log['Event'] == 'challenged').sum())
    npz_licks = int(np.nansum(behav_data['licks']))

    distance = vr_log['TotalRunDistance'].to_numpy(dtype=float)
    distance = distance[distance != -1]
    raw_distance = distance[-1] - distance[0]
    npz_distance = np.nanmax(behav_data['distance']) - np.nanmin(behav_data['distance'])

    print(f"{'metric':<26}{'raw (VR log)':<16}{'behaviour_data.npz':<20}")
    print(f"{'Rewards':<26}{n_vr_rewards:<16d}{npz_rewards:<20d}")
    print(f"{'VR challenged / licks':<26}{n_vr_licks:<16d}{npz_licks:<20d}  (informational only: npz licks "
          f"come from the NIDAQ lick channel, and 'challenged' events are not one per lick)")
    print(f"{'Distance run (cm)':<26}{raw_distance:<16.1f}{npz_distance:<20.1f}")

    if n_vr_rewards != npz_rewards:
        _flag(flags, f"VR log has {n_vr_rewards} rewards but behaviour_data.npz has {npz_rewards} -- "
              f"rewards outside the imaged frames, or a reward-channel/VR-log mismatch.")
    if abs(raw_distance - npz_distance) > 0.05 * max(abs(raw_distance), 1):
        _flag(flags, f"Distance run differs by more than 5% between the VR log ({raw_distance:.1f} cm) "
              f"and npz ({npz_distance:.1f} cm) -- imaging may not cover the whole VR session.")
    return flags


def verify(output_session_path, suite2p_plane_path, built=None):
    """Run all behaviour_data.npz checks for a cohort2 session. Returns a list of flag strings.

    :param built: build_behaviour_data_cohort2.build_behaviour_data()'s return value; the
        checks that need the raw re-derivation are skipped when it's None (build step skipped).
    """
    output_session_path = Path(output_session_path)

    print("--- Shapes ---")
    flags = check_shapes(output_session_path, suite2p_plane_path)

    if built is not None:
        behav_data = np.load(output_session_path / 'behaviour_data.npz')
        print("\n--- Frame overlap ---")
        flags += check_frame_overlap(built)
        print("\n--- Raw vs npz event counts ---")
        flags += check_event_counts(built, behav_data)
        print("\n--- VR log vs npz ---")
        flags += check_vr_metrics(built, behav_data)
    else:
        print("\n(skipping frame-overlap/event-count/VR-metric checks -- behaviour_data.npz build "
              "was skipped; pass force=True to redo it.)")

    print("\n--- Ca_traces identity ---")
    flags += check_ca_traces_identity(output_session_path, suite2p_plane_path)

    print(f"\n{len(flags)} issue(s) flagged." if flags else "\nNo issues flagged.")
    return flags
