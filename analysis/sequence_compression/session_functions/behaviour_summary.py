"""Across-session behavioural summaries used by sequence-compression notebooks."""

from collections import defaultdict
from itertools import combinations
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import plotting, utils


def _require_output_context(mouse_id, save_root):
    if mouse_id is None or save_root is None:
        raise ValueError(
            "mouse_id and save_root are required when plots are saved"
        )


def get_performance_per_stim_by_metric(Performances, Behaviours, metric='distance', remove_disengagement=False, plot_disengagement=False, include_A=False, condition='abbb', plot=True, *, mouse_id=None, save_root=None):
    '''Plots lick rate by distance or by time elapsed from the previous A'''
    
    n_sessions = len(Performances)

    # Collect performance data
    all_data = defaultdict(lambda: {"dist": [], "lick": []})

    for s, session_id in enumerate(Performances.keys()):
        if metric == 'time':
            A_A_diff, A_B_diff, licked_As, licked_Bs, _, _ = Performances[session_id].calc_time_from_A_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)

        elif metric == 'distance':
            A_A_diff, A_B_diff, licked_As, licked_Bs, _ = Performances[session_id].calc_distance_from_A_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
        
        elif metric == 'distance_rew':
            A_A_diff, A_B_diff, licked_As, licked_Bs, _ = Performances[session_id].calc_distance_from_rew_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
            
        elif metric == 'distance_A1rew':
            A_A_diff, A_B_diff, licked_As, licked_Bs, _ = Performances[session_id].calc_distance_from_A1rew_lick_rate(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
            if len(A_A_diff) == 2:
                A_A_diff = A_A_diff[::-1]
                licked_As = licked_As[::-1]
                
        elif metric == 'time_rew':
            A_A_diff, A_B_diff, licked_As, licked_Bs, _, _ = Performances[session_id].calc_time_from_rew_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)

        elif metric == 'distance_prob':
            A_A_diff, A_B_diff, licked_lms, _ = Performances[session_id].calc_distance_from_rew_lick_frac(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
        
        elif metric == 'lick_persistence': 
            A_A_diff, A_B_diff, licked_As, licked_Bs, _ = Behaviours[session_id].get_lick_persistence(plot=False, include_A=include_A, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
            
        # Collect data across sessions
        # --- Collect A data: one array per session, all under "A" ---
        if include_A and A_A_diff is not None:
            all_data["A"]["dist"].append(np.asarray(A_A_diff, dtype=float).ravel())
            all_data["A"]["lick"].append(np.asarray(licked_As, dtype=float).ravel())

        # --- Collect B data: one array per B type ---
        for i, (b_dist, b_lick) in enumerate(zip(A_B_diff, licked_Bs)):

            if len(A_B_diff) == 1:
                key = "B"
            else:
                key = f"B{i + 1}"

            all_data[key]["dist"].append(np.asarray(b_dist).ravel())
            all_data[key]["lick"].append(np.asarray(b_lick).ravel())

    # Concatenate across sessions
    for k in all_data:
        if len(all_data[k]["dist"]) > 0:
            all_data[k]["dist"] = np.concatenate(all_data[k]["dist"])
            all_data[k]["lick"] = np.concatenate(all_data[k]["lick"])
        else:
            all_data[k]["dist"] = np.array([])
            all_data[k]["lick"] = np.array([])
            
    all_distances = np.concatenate([all_data[k]["dist"] for k in all_data if len(all_data[k]["dist"]) > 0])
    bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

    results = {}

    for k in all_data:
        if len(all_data[k]["dist"]) == 0:
            continue

        c, m, s = utils.compute_binned_lick_rate(
            all_data[k]["dist"],
            all_data[k]["lick"],
            bins
        )
        results[k] = (c, m, s)

    # Plot
    if plot:
        _require_output_context(mouse_id, save_root)
        with mpl.rc_context({
            'axes.titlesize': 18,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 18,
        }):
            fig = plt.figure(figsize=(6,4))
            
            colors = {
                "A": "darkblue",
                "A1": "darkblue",
                "A2": "blue",
                "B": "orange",
                "B1": "orange",
                "B2": "gold",
                "B3": "brown"
            }

            for k, (c, m, s) in results.items():
                plt.errorbar(c, m, yerr=s, label=k, color=colors.get(k, None), marker='o')

            if metric == 'time':
                plt.xlabel('Time from A')
            elif metric == 'distance':
                plt.xlabel('Distance from A')
            elif metric == 'distance_rew' or metric == 'distance_prob' or metric == 'lick_persistence':
                plt.xlabel('Distance from A (rewarded)')
            elif metric == 'distance_A1rew':
                plt.xlabel('Distance from A1 (rewarded)')
            elif metric == 'time_rew':
                plt.xlabel('Time from A (rewarded)')
            
            if metric == 'distance_prob':
                plt.ylabel('Fraction of trials first licked')
            elif metric == 'lick_persistence':
                plt.ylabel('Mean lick count')
            else:
                plt.ylabel('Lick rate')

            ax = plt.gca()
            xmin = np.round(np.min(all_distances))
            xmax = np.round(np.max(all_distances))
            ax.set_xticks([xmin, xmax])
            if metric != 'lick_persistence':
                plt.ylim([0,1.1])
                plt.yticks([0,0.5,1])
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            legend = ax.legend(frameon=False, handlelength=0, handletextpad=0, markerscale=0)
            for handle in legend.legend_handles:
                handle.set_visible(False)
            for text in legend.get_texts():
                label = text.get_text()  
                text.set_color(colors.get(label, colors[label]))
                text.set_fontsize(24)

        plt.tight_layout()
        
        avg_save_path = os.path.join(save_root, f'sub-{mouse_id}', 'analysis', f'{condition}_sessions')
        if not os.path.exists(avg_save_path):
            os.makedirs(avg_save_path)

        if 'rew' in metric:
            anchor = 'rew'
        else:
            anchor = 'A'

        if remove_disengagement:
            if metric == 'lick_persistence' and include_A == True:
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_splitAB_incA_filtered.png')
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_splitAB_incA_filtered.svg', format="svg", bbox_inches="tight")
            else:
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_lick_rate_splitAB_filtered.png')
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_lick_rate_splitAB_filtered.svg', format="svg", bbox_inches="tight")
        else:
            if metric == 'lick_persistence' and include_A == False:
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_splitAB_incA.png')
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_splitAB_incA.svg', format="svg", bbox_inches="tight")
            else:
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_lick_rate_splitAB.png')
                plt.savefig(avg_save_path + f'/{metric}_from_{anchor}_lick_rate_splitAB.svg', format="svg", bbox_inches="tight")

        fig.suptitle(f'{utils.format_condition_label(cond=condition)} sub-{mouse_id} $\\mathit{{(n={n_sessions})}}$', fontsize=18)
        plt.show()

    return all_data

def get_p_lick_per_stim(Performances, condition='abbb', remove_disengagement=False, plot=True, *, mouse_id=None, save_root=None):
    """
    Aggregate across sessions and plot:

    1. Probability of first lick at each landmark type per distance bin from **rewarded** A
       (probabilities sum to 1 within each distance bin)
    2. Probability of first lick at each available landmark per landmark type per distance bin from **rewarded** A
    3. Number of first licks per landmark type per distance bin
    4. Number of trials per landmark type per distance bin
    5. Number of available trials (where no other lm was licked before) per landmark type per distance bin
    """
    
    # Collect per-session outputs
    all_A_A_diff = []
    all_A_B_diff = []
    all_lm_licked = []
    all_distances = []

    for session_id in Performances.keys():

        A_A_diff, A_B_diff, lm_licked, _, _, _, _, _, _, _, _ = Performances[session_id].calc_distance_from_rew_p_lick(plot=False, remove_disengagement=remove_disengagement)
        
        all_lm_licked.append(lm_licked)

        # Save raw distances to define common x-axis range
        all_A_A_diff.append(A_A_diff)
        all_A_B_diff.append(A_B_diff)

        all_B = np.concatenate(A_B_diff) if len(A_B_diff) > 0 else np.array([])
        session_distances = np.concatenate([A_A_diff, all_B.flatten()])
        all_distances.append(session_distances)

    # Concatenate per-session arrays
    all_lm_licked = np.concatenate(all_lm_licked)
    all_distances = np.concatenate(all_distances)

    num_Bs = len(all_A_B_diff[0])
    all_A_B_diff_concat = [
        np.concatenate([sess_Bs[i] for sess_Bs in all_A_B_diff])
        for i in range(num_Bs)
    ]
    all_A_A_diff = np.concatenate(all_A_A_diff)
     
    # Calculate common bins across sessions
    bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_idx = []
    for i in range(num_Bs):
        bin_idx.append(np.digitize(all_A_B_diff_concat[i], bins))
    bin_idx.append(np.digitize(all_A_A_diff, bins))
    
    # Compute distance bin, available distance bin, and lick counts  
    num_lms = num_Bs + 1 

    all_distance_bin_counts = np.array([
        [np.sum(bin_idx[i] == b) for b in range(1, len(bins))] for i in range(num_lms)
    ])
    
    all_avail_distance_bin_counts = np.array([
        [np.sum((bin_idx[i] == b) & (~np.any(all_lm_licked[:, :i], axis=1))) 
            for b in range(1, len(bins))] for i in range(num_lms)], dtype=int)

    all_lm_lick_counts = np.array([
        [np.sum(all_lm_licked[:,i][bin_idx[i] == b])
         for b in range(1, len(bins))] for i in range(num_lms)], dtype=int)
  
    # Total licks per bin across landmark types
    total_lick_counts = np.sum(all_lm_lick_counts, axis=0)

    # Probability of each landmark type within each bin
    lm_lick_prob = np.divide(
        all_lm_lick_counts,
        total_lick_counts,
        out=np.full_like(all_lm_lick_counts, np.nan, dtype=float),
        where=total_lick_counts > 0
    )

    # Probability of lick within each bin for available (no attempt before) landmarks 
    avail_lm_lick_prob = np.divide(
        all_lm_lick_counts,
        all_avail_distance_bin_counts,
        out=np.full_like(all_avail_distance_bin_counts, np.nan, dtype=float),
        where=all_avail_distance_bin_counts > 0
    )

    if plot:
        _require_output_context(mouse_id, save_root)
        # 1. Probability of first lick
        fig1 = plotting.plot_data(bin_centers, lm_lick_prob, all_distances, 'Probability of first lick')

        # 2. Probability of first lick if lm is available
        fig2 = plotting.plot_data(bin_centers, avail_lm_lick_prob, all_distances, 'Probability of first lick\nin available landmark')

        # 3. Number of first licks
        fig3 = plotting.plot_data(bin_centers, all_lm_lick_counts, all_distances, 'Number of first licks')

        # 4. Number of trials
        fig4 = plotting.plot_data(bin_centers, all_distance_bin_counts, all_distances, 'Number of trials')

        # 5. Number of available trials
        fig5 = plotting.plot_data(bin_centers, all_avail_distance_bin_counts, all_distances, 'Number of available trials')

        for fig in [fig1, fig2, fig3, fig4, fig5]:
            fig.tight_layout()
            # fig.suptitle(f'{utils.format_condition_label(cond=condition)} sub-{mouse_id} $\\mathit{{(n={len(sessions)})}}$', fontsize=18)
        
        # Save figures
        avg_save_path = os.path.join(save_root, f'sub-{mouse_id}', 'analysis', f'{condition}_sessions')
        if not os.path.exists(avg_save_path):
            os.makedirs(avg_save_path, exist_ok=True)
        
        fig1.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_probability.png'))
        fig2.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_probability_avail_lm.png'))
        fig3.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_counts.png'))
        fig4.savefig(os.path.join(avg_save_path, f'distance_from_rew_trial_counts.png'))
        fig5.savefig(os.path.join(avg_save_path, f'distance_from_rew_avail_trial_counts.png'))

        fig1.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_probability.svg'), format="svg", bbox_inches="tight")
        fig2.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_probability_avail_lm.svg'), format="svg", bbox_inches="tight")
        fig3.savefig(os.path.join(avg_save_path, f'distance_from_rew_first_lick_counts.svg'), format="svg", bbox_inches="tight")
        fig4.savefig(os.path.join(avg_save_path, f'distance_from_rew_trial_counts.svg'), format="svg", bbox_inches="tight")
        fig5.savefig(os.path.join(avg_save_path, f'distance_from_rew_avail_trial_counts.svg'), format="svg", bbox_inches="tight")

        for fig in [fig1, fig2, fig3, fig4, fig5]:
            fig.suptitle(f'{utils.format_condition_label(cond=condition)} sub-{mouse_id} $\\mathit{{(n={len(Performances)})}}$', fontsize=18)
    
        return (
            all_A_A_diff, all_A_B_diff_concat, bin_centers, all_lm_licked, lm_lick_prob, avail_lm_lick_prob,
            all_lm_lick_counts, all_distance_bin_counts, all_avail_distance_bin_counts,
            fig1, fig2, fig3, fig4, fig5)
    
    else:
        return (
            all_A_A_diff, all_A_B_diff_concat, bin_centers, all_lm_licked, lm_lick_prob, avail_lm_lick_prob,
            all_lm_lick_counts, all_distance_bin_counts, all_avail_distance_bin_counts, 
            None, None, None, None, None)

def get_lick_counts_sessions(sessions, data, settings, misses=False, omissions=False, condition='abbb', subcondition=None, require_other_a_reward=False, calculate_distances=None, *, mouse_id=None, save_root=None, parse_functions=None, save_plot=True):
    '''Calculate and plot lick counts for AAB and periodic conditions (e.g. ABAB and ABB).

    Distance-from-rewarded-A plots are optional and disabled by default for
    omission trials, where a rewarded reference A may be absent.
    Returns a raw table with one row per included landmark and a summary
    table with the count, mean, and sample SD for each landmark type.
    '''

    _require_output_context(mouse_id, save_root)

    if parse_functions is None:
        from preprocessing import (
            parse_session_functions_cohort1_sc as parse_functions
        )

    if calculate_distances is None:
        calculate_distances = not omissions

    # Collect data
    all_data = defaultdict(lambda: {"dist": [], "lick": [], "counts": []})
    count_rows = []

    for s, sess in enumerate(sessions):
        session_id = utils.extract_session_id(sess)  

        if misses:
            lick_counter, licked_As, licked_Bs, A_A_diff, A_B_diff, _ = parse_functions.get_lick_counts(data[session_id], settings[session_id], plot=False, threshold='above', misses=misses, omissions=omissions, require_other_a_reward=require_other_a_reward, calculate_distances=calculate_distances, exclude_b_after_invalid_a=not omissions)
        else:
            lick_counter, licked_As, licked_Bs, A_A_diff, A_B_diff, _ = parse_functions.get_lick_counts(data[session_id], settings[session_id], plot=False, threshold='below', misses=misses, omissions=omissions, require_other_a_reward=require_other_a_reward, calculate_distances=calculate_distances, exclude_b_after_invalid_a=not omissions)
   
        # Collect data across sessions
        for landmark_type, counts in lick_counter.items():
            counts = np.asarray(counts, dtype=float)
            all_data[landmark_type]["counts"].append(counts)
            for lick_count in counts[np.isfinite(counts)]:
                count_rows.append({
                    "mouse_id": mouse_id,
                    "session_id": session_id,
                    "condition": condition,
                    "selection": "miss" if misses else "omission" if omissions else "all",
                    "landmark_type": landmark_type,
                    "lick_count": lick_count,
                })

        if calculate_distances:
            num_As = len(licked_As)
            A_order = [1, 0] if num_As == 2 else range(num_As)
            for i, A_val in enumerate(A_order):
                key = f"A{i+1}" if len(A_A_diff) > 1 else "A"
                all_data[key]["dist"].append(A_A_diff[A_val])
                all_data[key]["lick"].append(licked_As[A_val])

            for i in range(len(A_B_diff)):
                key = f"B{i+1}" if len(A_B_diff) > 1 else "B"
                all_data[key]["dist"].append(A_B_diff[i])
                all_data[key]["lick"].append(licked_Bs[i])

    # Concatenate across sessions
    for k in all_data:
        if len(all_data[k]["dist"]) > 0:
            all_data[k]["dist"] = np.concatenate(all_data[k]["dist"])
            all_data[k]["lick"] = np.concatenate(all_data[k]["lick"])
        else:
            all_data[k]["dist"] = np.array([])
            all_data[k]["lick"] = np.array([])
        
        if len(all_data[k]["counts"]) > 0:
            all_data[k]["counts"] = np.concatenate(all_data[k]["counts"])
        else:
            all_data[k]["counts"] = np.array([])
            
    results = {}
    bins = None
    distance_arrays = [all_data[k]["dist"] for k in all_data if len(all_data[k]["dist"]) > 0]
    if distance_arrays:
        all_distances = np.concatenate(distance_arrays)
        bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

        for k in all_data:
            if len(all_data[k]["dist"]) == 0:
                continue
            c, m, s = utils.compute_binned_lick_rate(
                all_data[k]["dist"], all_data[k]["lick"], bins
            )
            results[k] = (c, m, s)

    
    # Stats 
    ttest_results = {}

    for k1, k2 in combinations(all_data.keys(), 2):
        x = all_data[k1]["counts"]
        y = all_data[k2]["counts"]
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        # Skip empty groups
        if len(x) == 0 or len(y) == 0:
            continue

        # Welch's t-test (recommended unless equal variances are assumed)
        t_stat, p_val = ttest_ind(x, y, equal_var=False, nan_policy="omit", alternative='two-sided')

        ttest_results[(k1, k2)] = {
            "t": t_stat,
            "p": p_val,
            "n1": len(x),
            "n2": len(y),
        }

    pairs = list(ttest_results.keys())
    pvals = [ttest_results[p]["p"] for p in pairs]

    if pvals:
        reject, pvals_corr, _, _ = multipletests(pvals, method="holm")
    else:
        reject, pvals_corr = [], []

    # Print results
    for (k1, k2), res in ttest_results.items():
        print(
            f"{k1} vs {k2}: "
            f"t = {res['t']:.3f}, "
            f"p = {res['p']:.4g}, "
            f"n = ({res['n1']}, {res['n2']})"
        )

    corrected_results = {}
    for pair, p_raw, p_corr, sig in zip(pairs, pvals, pvals_corr, reject):
        corrected_results[pair] = {
            "p": p_corr,
            "significant": sig
        }
        print(f"{pair}: raw p={p_raw:.4g}, corrected p={p_corr:.4g}, significant={sig}")
    
    def p_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"
        
    # Plot
    with mpl.rc_context({
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 18,
    }):
        fig1, ax1 = plt.subplots(figsize=(6,4))
        if results:
            fig2, ax2 = plt.subplots(figsize=(6,4))
        else:
            fig2 = None
            ax2 = None
        
        colors = {
            "A": ["darkblue", "mediumblue"],
            "A1": ["darkblue", "mediumblue"],
            "A2": ["blue", "dodgerblue"],
            "B": ["orange", "gold"],
            "B1": ["orange", "gold"],
            "B2": ["gold", "yellow"],
            "B3": ["brown", "sandybrown"]
        }

        # 1. Bar plot of number of licks per lm type
        labels = [k for k in all_data if np.any(np.isfinite(all_data[k]["counts"]))]
        if not labels:
            raise ValueError(f"No lick counts were found for {condition} ({'omissions' if omissions else 'misses' if misses else 'all landmarks'}).")
        values = [all_data[k]["counts"] for k in labels]
        means = [np.nanmean(all_data[k]["counts"]) for k in labels]

        bar_colors = [colors[l][0] for l in labels]
        dot_colors = [colors[l][1] for l in labels]

        x = np.arange(len(labels))
        ax1.bar(x, means, color=bar_colors)

        # Individual data points
        for i, v in enumerate(values):
            jitter = np.random.uniform(-0.08, 0.08, size=len(v))
            ax1.scatter(
                np.full(len(v), x[i]) + jitter,
                v,
                color=dot_colors[i],
                alpha=0.5
            )

        # Plot significance stars
        y_max = max(np.nanmax(v) for v in values) # height of tallest point

        # Starting height for significance bars
        step = max(y_max * 0.1, 0.5)
        height = y_max + step * 0.2

        for pair, res in corrected_results.items():

            k1, k2 = pair
            x1 = labels.index(k1)
            x2 = labels.index(k2)

            stars = p_to_stars(res["p"])

            ax1.plot(
                [x1, x1, x2, x2],
                [height, height + step/4, height + step/4, height],
                color='black',
                lw=1.5
            )

            ax1.text(
                (x1 + x2)/2,
                height + step/5,
                stars,
                ha='center',
                va='bottom',
                fontsize=12
            )

            height += step

        ax1.set_xticks(x)
        if misses:
            ax1.set_xticklabels(f"{label} (miss)" if label.startswith("A") else label for label in labels)
        if omissions:
            ax1.set_xticklabels(f"{label}\n(omitted)" if label.startswith("A") else label for label in labels)
        if not misses and not omissions:
            ax1.set_xticklabels(labels)
        ax1.set_ylabel("Licks per landmark")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim(top=height + step)

        # 2. Lick counts as a function of distance from rewarded A1s
        if fig2 is not None:
            for k, (c, m, s) in results.items():
                display_label = (f"{k} (miss)" if misses else f"{k} (omitted)" if omissions else k)
                ax2.errorbar(c, m, yerr=s, label=display_label, color=colors[k][0], marker='o')

                ymin, ymax = ax2.get_ylim()
                ax2.set_yticks([0, np.round(ymax)])
                ax2.set_xticks([bins[0], bins[-1]])

                legend = ax2.legend(frameon=False, loc='upper right', handlelength=0, handletextpad=0, markerscale=0)
                for handle, text in zip(legend.legend_handles, legend.get_texts()):
                    text.set_color(handle.get_color())
                    handle.set_visible(False)
                    # text.set_fontsize(24)
                
                ax2.set_xlabel('Distance A1 (rewarded) → ')
                ax2.set_ylabel('Lick counts')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
    
        plt.tight_layout()
        
        session_group = f'{condition}_{subcondition}_sessions' if subcondition else f'{condition}_sessions'
        avg_save_path = os.path.join(save_root, f'sub-{mouse_id}', 'analysis', session_group)
        if not os.path.exists(avg_save_path):
            os.makedirs(avg_save_path)

        suffix = "_misses" if misses else "_omissions" if omissions else ""
        base1 = os.path.join(avg_save_path, f"lick_counts{suffix}")
        base2 = os.path.join(avg_save_path, f"lick_counts_A1_distance{suffix}")
        count_table = pd.DataFrame(count_rows)
        summary_group_columns = [
            'mouse_id', 'condition', 'selection', 'landmark_type'
        ]
        summary_table = (
            count_table
            .groupby(summary_group_columns, as_index=False)
            .agg(
                n_landmarks=('lick_count', 'count'),
                mean_lick_count=('lick_count', 'mean'),
                std_lick_count=('lick_count', 'std'),
            )
        )
        count_table.to_csv(f"{base1}.csv", index=False)
        summary_table.to_csv(f"{base1}_summary.csv", index=False)

        # save fig1 (always exists)
        if save_plot:
            fig1.savefig(f"{base1}.png")
            fig1.savefig(f"{base1}.svg", format="svg", bbox_inches="tight")
            fig1.suptitle(f'{utils.format_condition_label(cond=condition)} sub-{mouse_id} $\\mathit{{(n={len(sessions)})}}$', fontsize=18)

            # save fig2 only if it exists
            if fig2 is not None:
                fig2.savefig(f"{base2}.png")
                fig2.savefig(f"{base2}.svg", format="svg", bbox_inches="tight")
                fig2.suptitle(f'{utils.format_condition_label(cond=condition)} sub-{mouse_id} $\\mathit{{(n={len(sessions)})}}$', fontsize=18)
        
        plt.show()

    return count_table, summary_table
