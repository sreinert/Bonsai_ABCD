"""Goal-progress tuning for GoalProgress/cohort3 sessions, sequence and shaping alike.

`cellTV_functions.py`'s existing `extract_goal_progress`/`calc_goal_tuningix` only support
exactly 3 or 4 goals, and identify segments by raw ordinal reward-cycling (segment k belongs
to goal k mod ngoals) rather than by verifying the mouse's next lick actually landed on the
correct next goal. The "correct transitions" variants that DO verify this
(`extract_gp_correct_transitions[_unsorted]`, `calc_gp_tuning_corr_trans`) depend on
`session['correct_transitions']`, which is populated by
`preprocessing/parse_session_functions.py::get_corr_transitions` -- but that function (and
the whole generic pipeline it lives in) is a different, older/parallel pipeline from
`preprocessing/parse_session_functions_cohort3.py`'s `analyse_npz_sequences`, which is what
GoalProgress/cohort3 sessions (and musicboxes/session_overview.py) actually use.
`session['correct_transitions']` does not exist there.

This module is a deliberately SMALL, targeted addition (not a full copy of
cellTV_functions.py -- see below for why) containing:

  - get_corr_transitions_cohort3(session): cohort3-compatible port of
    parse_session_functions.py's get_corr_transitions, for sessions with > 1 rewarded
    landmark (a real sequence task, e.g. A->B->C->D). "Correct" is defined identically to
    the original: the mouse's very next lick after a reward at `goal` must land on the next
    goal in the fixed cycle, with zero tolerance for an intervening wrong-landmark lick.

  - get_shaping_transitions(session): the shaping equivalent for sessions with exactly 1
    rewarded landmark (no meaningful "next goal identity" to verify). A transition is any
    pair of landmark VISITS that are consecutive in visit order (nothing unrewarded landmark
    visited in between) and were BOTH rewarded.

  Both populate session['correct_transitions'] as a list of (goal, next_goal, start_frame,
  end_frame) tuples, using the SAME 4-column shape so the per-cell functions below don't
  need to know which one ran.

  - extract_goal_progress_pooled(dF, cell, session, bins=90, shuffle=False): bins dF/F
    within every session['correct_transitions'] segment, pooled across ALL segment types
    together into one (n_transitions, bins) matrix. Mirrors
    cellTV_functions.py's extract_gp_correct_transitions_unsorted. Used for BOTH shaping
    (its only mode) and sequence's "collected across all segments" mode.

  - extract_goal_progress_by_segment(dF, cell, session, bins=90, shuffle=False): sequence
    sessions only -- same binning, but split by segment TYPE (starting goal identity)
    instead of pooled. Mirrors extract_gp_correct_transitions. Returns
    {goal_id: (n_trials_of_that_type, bins) array}.

  - calc_goal_progress_tuning(dF, cell, session, bins=90): per-cell tuning index
    ((max-min)/mean of the pooled average curve) and preferred phase (bin of peak firing).
    Mirrors calc_gp_tuning_corr_trans.

Why a small new module instead of a literal copy of cellTV_functions.py: the two
"correct-transitions" functions above are genuinely new (nothing to fork from), while the
per-cell binning/tuning functions are close enough to existing cellTV_functions.py code
that duplicating the whole ~1500-line file would just create two copies of unrelated shared
helpers (extract_cell_trace, extract_lick_rate, plotting helpers, position/reward tuning,
...) to keep in sync. This module imports what it needs from cellTV_functions directly
instead. Flag if you'd rather have a literal full-file copy.
"""

import numpy as np


def extract_cell_trace(dF, cell, plot=False, session=None, frame_range=None):
    """Minimal trace accessor used by the goal-progress calculations."""
    return np.asarray(dF)[cell]


# ---------------------------------------------------------------------------
# Session-level: find goal-progress transitions (session['correct_transitions'])
# ---------------------------------------------------------------------------

def get_corr_transitions_cohort3(session, correct_only=True):
    """Sequence-session (>1 rewarded landmark) version of finding transitions.

    Adapted from preprocessing/parse_session_functions.py::get_corr_transitions for
    cohort3's session-struct field layout:
      - session['all_lms']    : per-visit landmark-slot identity (0-indexed), the
                                 per-lap-concatenated sequence of every landmark visited.
      - session['lm_idx']     : per-FRAME landmark-visit-event id (1-indexed, 0 = none) --
                                 lm_idx[t] == i+1 means frame t falls inside visit i.
      - session['licked_lms']/['rewarded_lms'] : FLAT boolean arrays already aligned to
                                 all_lms (from get_licks_rewards_npz) -- unlike the generic
                                 pipeline's 2D (laps x landmarks) arrays that the original
                                 get_corr_transitions itself has to flatten first.
      - session['goal_ids']   : list of goal landmark-slot identities, in fixed cycle
                                 order (length == ses_settings['availableRewardPositions']).

    Because all_lms/lm_idx already share one consistent 0-indexed "visit number" (visit i
    corresponds to lm_idx == i+1, confirmed by get_licks_rewards_npz's own indexing), this
    can match a rewarded visit directly to its reward frame instead of reconstructing
    per-lap offsets the way the original (differently-indexed) get_corr_transitions does.

    :param correct_only: if True (default), only include a transition when the mouse's
        very next lick after the reward lands on the correct next goal (zero tolerance for
        an intervening wrong-landmark lick) -- the original "correct transitions" behaviour.
        If False, include every reward-to-reward interval for each goal's reward
        occurrences regardless of what the mouse did in between, still labeled by the
        expected (goal, next_goal) identity pair from the fixed cycle order.

    Populates and returns session with session['correct_transitions'] = a list of
    (goal, next_goal, start_rew_frame, end_rew_frame) tuples.
    """
    licked_lm_ix = np.where(session['licked_lms'] == 1)[0]
    reward_lm_vector = session['rewarded_lms']
    lm_sequence = session['all_lms']
    goal_values = session['goal_ids']

    # 0-indexed visit id for every recorded reward frame (see docstring above).
    reward_visit_id = session['lm_idx'][session['rewards']] - 1

    correct_transitions = []
    for g, goal in enumerate(goal_values):
        nth_reward = np.intersect1d(np.where(reward_lm_vector == 1)[0], np.where(lm_sequence == goal)[0])
        next_goal = goal_values[g + 1] if g + 1 < len(goal_values) else goal_values[0]
        for i in range(len(nth_reward) - 1):
            if correct_only:
                after = licked_lm_ix[licked_lm_ix > nth_reward[i]]
                if len(after) == 0:
                    continue
                next_lm = lm_sequence[after[0]]
                if next_lm != next_goal:
                    continue
            else:
                next_lm = next_goal  # unverified: label with the expected next goal anyway

            start_matches = np.where(reward_visit_id == nth_reward[i])[0]
            if len(start_matches) == 0:
                continue
            start_ix = start_matches[0]
            start_rew_frame = session['rewards'][start_ix]
            end_rew_frame = (session['rewards'][start_ix + 1] if start_ix + 1 < len(session['rewards'])
                             else session['rewards'][-1])
            correct_transitions.append((goal, next_lm, start_rew_frame, end_rew_frame))

    session['correct_transitions'] = correct_transitions
    return session


def get_shaping_transitions(session, correct_only=True):
    """Shaping-session (exactly 1 rewarded landmark) version of finding transitions.

    There's only one goal identity, so "did the next lick land on the correct next goal"
    isn't meaningful.

    :param correct_only: if True (default), a transition is any pair of landmark VISITS
        that are consecutive in visit order (i.e. nothing else was visited in between) and
        were BOTH rewarded -- this excludes cases where an unrewarded landmark was visited
        between two reward events (e.g. during lenient early-training regimes where other
        landmarks can occasionally also deliver reward). If False, every reward-to-reward
        interval in session['rewards'] is included unconditionally (no visit-adjacency
        check), which for a single-goal session is simply every consecutive pair of reward
        events regardless of what happened in between.

    Populates and returns session with session['correct_transitions'] = a list of
    (goal, goal, start_rew_frame, end_rew_frame) tuples -- goal repeated in both columns
    (there's only one) so this is interchangeable with get_corr_transitions_cohort3's
    output for the per-cell functions below.
    """
    if len(session['goal_ids']) != 1:
        raise ValueError(f"get_shaping_transitions expects exactly 1 goal, "
                          f"got {len(session['goal_ids'])} (session['goal_ids']={session['goal_ids']}) "
                          f"-- use get_corr_transitions_cohort3 for sequence sessions instead.")
    goal = session['goal_ids'][0]
    rewarded_lms = session['rewarded_lms']
    reward_visit_id = session['lm_idx'][session['rewards']] - 1
    rewarded_visits = np.where(rewarded_lms == 1)[0]

    if correct_only:
        rewarded_set = set(rewarded_visits.tolist())
        keep_visits = [i for i in rewarded_visits if (i + 1) in rewarded_set]
    else:
        keep_visits = rewarded_visits[:-1] if len(rewarded_visits) > 1 else []

    correct_transitions = []
    for i in keep_visits:
        start_matches = np.where(reward_visit_id == i)[0]
        if len(start_matches) == 0:
            continue
        start_ix = start_matches[0]
        start_rew_frame = session['rewards'][start_ix]
        end_rew_frame = (session['rewards'][start_ix + 1] if start_ix + 1 < len(session['rewards'])
                         else session['rewards'][-1])
        correct_transitions.append((goal, goal, start_rew_frame, end_rew_frame))

    session['correct_transitions'] = correct_transitions
    return session


def get_goal_progress_transitions(session, correct_only=True):
    """Convenience dispatcher: picks get_corr_transitions_cohort3 (sequence) or
    get_shaping_transitions (shaping) based on len(session['goal_ids']), matching the
    ses_settings['availableRewardPositions'] > 1 vs == 1 distinction.

    :param correct_only: see get_corr_transitions_cohort3/get_shaping_transitions -- if
        False, use every reward-to-reward segment unconditionally instead of only ones
        verified "correct" by the respective criterion.
    """
    if len(session['goal_ids']) > 1:
        return get_corr_transitions_cohort3(session, correct_only=correct_only)
    return get_shaping_transitions(session, correct_only=correct_only)


# ---------------------------------------------------------------------------
# Per-cell: bin dF/F within transitions, and compute a tuning index/preferred phase
# ---------------------------------------------------------------------------

def extract_goal_progress_pooled(dF, cell, session, bins=90, shuffle=False):
    """Bin dF/F within every session['correct_transitions'] segment, pooled across ALL
    segment types into one (n_transitions, bins) matrix. Mirrors cellTV_functions.py's
    extract_gp_correct_transitions_unsorted -- works for both sequence and shaping
    transition lists, since both share the same (goal, next_goal, start, end) shape.
    """
    transitions = session['correct_transitions']
    if len(transitions) == 0:
        raise ValueError("session['correct_transitions'] is empty -- no valid goal-progress "
                          "segments found for this session.")
    transitions = np.asarray(transitions)
    start_ix = transitions[:, 2].astype(int)
    end_ix = transitions[:, 3].astype(int)

    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    binned = np.full((len(start_ix), bins), np.nan)
    for i in range(len(start_ix)):
        if end_ix[i] <= start_ix[i]:
            continue
        frames = np.arange(start_ix[i], end_ix[i])
        trace = dF_cell[frames]
        if shuffle:
            trace = trace.copy()
            np.random.shuffle(trace)
        bin_edges = np.linspace(start_ix[i], end_ix[i], bins + 1)
        bin_ix = np.digitize(frames, bin_edges)
        for j in range(bins):
            sel = trace[bin_ix == j + 1]
            if len(sel):
                binned[i, j] = np.mean(sel)
    return binned


def extract_goal_progress_by_segment(dF, cell, session, bins=90, shuffle=False):
    """Sequence sessions only: same binning as extract_goal_progress_pooled, but split by
    segment TYPE (starting goal identity) instead of pooled. Mirrors cellTV_functions.py's
    extract_gp_correct_transitions.

    Returns {goal_id: (n_trials_of_that_type, bins) array}.
    """
    transitions = np.asarray(session['correct_transitions'])
    goals = np.unique(transitions[:, 0].astype(int))
    per_segment = {}
    for goal in goals:
        segment_session = dict(session)  # shallow copy -- only swapping correct_transitions
        segment_session['correct_transitions'] = transitions[transitions[:, 0].astype(int) == goal]
        per_segment[int(goal)] = extract_goal_progress_pooled(dF, cell, segment_session, bins=bins, shuffle=shuffle)
    return per_segment


def calc_goal_progress_tuning(dF, cell, session, bins=90):
    """Per-cell tuning index and preferred phase from the POOLED goal-progress curve.
    Mirrors cellTV_functions.py's calc_gp_tuning_corr_trans.

    Returns (avg_bin, phase_pref, tuning_index):
      avg_bin      : (bins,) trial-averaged goal-progress curve.
      phase_pref   : bin index (0..bins-1) of peak firing.
      tuning_index : (max-min)/mean of avg_bin -- how strongly phase-modulated the cell is.
    """
    binned = extract_goal_progress_pooled(dF, cell, session, bins=bins, shuffle=False)
    avg_bin = np.nanmean(binned, axis=0)
    phase_pref = int(np.nanargmax(avg_bin))
    tuning_index = (np.nanmax(avg_bin) - np.nanmin(avg_bin)) / np.nanmean(avg_bin)
    return avg_bin, phase_pref, tuning_index


def calc_goal_progress_polar_curve(dF, cell, session, bins=90):
    """Per-cell goal-progress curve concatenated across every transition type in
    session['goal_ids']'s fixed cycle order (goal[0]->goal[1], goal[1]->goal[2], ...,
    goal[-1]->goal[0]), for polar plotting -- one `bins`-wide wedge per transition type.

    A transition type with zero entries in session['correct_transitions'] (e.g. because
    get_corr_transitions_cohort3 was run with correct_only=True and the mouse never
    completed that particular transition correctly) gets an all-zero wedge rather than
    being silently skipped, so every cell's polar plot always spans the full circle with
    the same number of goals/bins -- a flat-zero wedge is itself informative (this segment
    was never observed under the current criterion).

    Returns a 1D array of length bins * len(session['goal_ids']).
    """
    goal_ids = session['goal_ids']
    transitions = np.asarray(session['correct_transitions']) if len(session['correct_transitions']) else \
        np.empty((0, 4))
    curve = np.zeros(bins * len(goal_ids))
    if transitions.shape[0] == 0:
        return curve

    goal_col = transitions[:, 0].astype(int)
    for k, goal in enumerate(goal_ids):
        mask = goal_col == goal
        if not np.any(mask):
            continue  # leave this wedge at zero -- no valid transitions of this type
        segment_session = dict(session)  # shallow copy -- only swapping correct_transitions
        segment_session['correct_transitions'] = transitions[mask]
        binned = extract_goal_progress_pooled(dF, cell, segment_session, bins=bins)
        curve[k * bins:(k + 1) * bins] = np.nanmean(binned, axis=0)
    return curve
