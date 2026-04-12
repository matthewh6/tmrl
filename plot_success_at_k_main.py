"""
Main Success@K figure: BC vs PostBC vs CSP (best tcont auto-selected).

Usage:
    python plot_success_at_k_main.py \
        --json  results.json \
        --title "cube-single" \
        --out   success_at_k_main.pdf
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import shared_style
matplotlib.rcParams.update(shared_style.RC)

# Success@K uses muted colors — same hues as RL curves, quieter tone
# PostBC here = PostBC pre-training evaluated before RL,
# so it uses the same gray as PostBC in the sim results figure
METHODS = [
    dict(key_arg="csp",    label=shared_style.LABELS['CSP'],
         color=shared_style.COLORS_MUTED['CSP'],
         marker=shared_style.MARKERS['CSP'],         lw=2.0, alpha=1.0),
    dict(key_arg="postbc", label=shared_style.LABELS['PostBC'],
         color=shared_style.COLORS_MUTED['PostBC'],
         marker=shared_style.MARKERS['PostBC'], lw=1.4, alpha=0.9),
    dict(key_arg="dsrl",   label=shared_style.LABELS['BC'],
         color=shared_style.COLORS_MUTED['BC'],
         marker=shared_style.MARKERS['BC'],          lw=1.4, alpha=0.9),
]

MARKER_KS  = {1, 5, 10, 20}
TCONT_KEYS = ["tmrl_t0.00", "tmrl_t0.25", "tmrl_t0.50", "tmrl_t0.75", "tmrl_t1.00"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_curve(data, key):
    if key not in data:
        raise KeyError(f"'{key}' not in JSON. Available: {list(data.keys())}")
    pak   = data[key]["success_at_k"]
    k_max = max(int(k) for k in pak)
    arr   = np.zeros(k_max)
    for k_str, v in pak.items():
        arr[int(k_str) - 1] = v
    return arr


def auto_best_tcont(data):
    """Pick tcont with highest Success@K_max, break ties by AUC."""
    best_key, best_val = None, -1.0
    for key in TCONT_KEYS:
        if key not in data:
            continue
        curve = extract_curve(data, key)
        score = curve[-1] * 1000 + curve.sum()
        if score > best_val:
            best_val = score
            best_key = key
    return best_key


def load_npz(npz_path, key):
    if not npz_path or not os.path.exists(npz_path):
        return None
    data    = np.load(npz_path)
    arr_key = f"{key}_successes"
    if arr_key not in data:
        return None
    return data[arr_key].astype(bool)


def bootstrap_ci(successes, k_max, n_bootstrap=2000, ci=0.95):
    n_seeds = successes.shape[0]
    k_max   = min(k_max, successes.shape[1])
    rng     = np.random.default_rng(seed=0)
    boot    = np.zeros((n_bootstrap, k_max), dtype=np.float32)
    for b in range(n_bootstrap):
        idx    = rng.integers(0, n_seeds, size=n_seeds)
        sample = successes[idx]
        for k in range(1, k_max + 1):
            boot[b, k - 1] = sample[:, :k].any(axis=1).mean()
    alpha = (1.0 - ci) / 2.0
    return (np.percentile(boot, alpha * 100, axis=0),
            np.percentile(boot, (1 - alpha) * 100, axis=0))


def smart_ylim(curves, pad=0.02):
    ymax = max(c.max() for c in curves)
    # Round up to nearest 0.1 for clean ticks — avoids dense tick labels
    ymax_r = np.ceil(ymax / 0.1) * 0.1
    return -pad, min(ymax_r + pad, 1.0)


def clean_yticks(ymin, ymax):
    """Return ~4-6 evenly spaced y-ticks using the best 'nice' interval."""
    span = ymax - ymin
    if span <= 0:
        return [0.0]
    nice_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5]
    best = nice_steps[-1]
    for s in nice_steps:
        n = span / s
        if 3.5 <= n <= 6.5:
            best = s
            break
    ticks = np.arange(0.0, ymax + best * 0.5, best)
    return [t for t in ticks if ymin - 0.001 <= t <= ymax + 0.001]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",        required=True)
    parser.add_argument("--npz",         default=None)
    parser.add_argument("--best_tcont",  default=None)
    parser.add_argument("--title",       default=None)
    parser.add_argument("--out",         default="success_at_k_main.pdf")
    parser.add_argument("--no_shading",  action="store_true")
    parser.add_argument("--figsize",     nargs=2, type=float,
                        default=list(shared_style.FIGSIZE_SINGLE))
    parser.add_argument("--dpi",         type=int,   default=300)
    parser.add_argument("--ci",          type=float, default=0.95)
    parser.add_argument("--n_bootstrap", type=int,   default=2000)
    args = parser.parse_args()

    data       = load_json(args.json)
    best_tcont = args.best_tcont or auto_best_tcont(data)
    print(f"CSP tcont: {best_tcont}")

    sources = {"dsrl": "dsrl", "postbc": "postbc", "csp": best_tcont}

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    shading = not args.no_shading
    curves  = []

    for m in METHODS:
        json_key  = sources[m["key_arg"]]
        success_k    = extract_curve(data, json_key)
        k_max     = len(success_k)
        k_vals    = np.arange(1, k_max + 1)
        markevery = [i for i, k in enumerate(k_vals) if k in MARKER_KS]

        low = high = None
        if shading and args.npz:
            succ = load_npz(args.npz, json_key)
            if succ is not None:
                low, high = bootstrap_ci(succ, k_max=k_max,
                                         n_bootstrap=args.n_bootstrap,
                                         ci=args.ci)

        ax.plot(k_vals, success_k,
                color=m["color"], linestyle="-",
                marker=m["marker"], markevery=markevery,
                markersize=5, markeredgewidth=0.8, markeredgecolor="white",
                linewidth=m["lw"], alpha=m["alpha"],
                label=m["label"], zorder=3)

        if low is not None and high is not None:
            ax.fill_between(k_vals, low, high,
                            alpha=0.12, color=m["color"], zorder=2)
        curves.append(success_k)

    ymin, ymax = smart_ylim(curves)
    ax.set_ylim(ymin, ymax)
    yticks = clean_yticks(ymin, ymax)
    ax.set_yticks(yticks)
    if len(yticks) >= 2:
        step = yticks[1] - yticks[0]
        decimals = 0 if step >= 0.95 else (1 if step >= 0.095 else 2)
    else:
        decimals = 1
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.set_xlim(1, k_max + 0.3)
    ax.set_xlabel("$K$")
    ax.set_ylabel("Success@$K$")
    ax.set_xticks([k for k in [1, 5, 10, 20] if k <= k_max])

    # Title: use shared task label mapping if available
    title = args.title
    if title:
        title = shared_style.TASK_LABELS.get(title, title)
        ax.set_title(title, pad=4)

    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved -> {args.out}")

    ks = [1, 5, 10, 20]
    print(f"\n{'Method':<10}" + "".join(f"  Success@{k:>2}" for k in ks))
    print("-" * 42)
    for m, success_k in zip(METHODS, curves):
        row = f"{m['label']:<10}"
        for k in ks:
            row += f"  {success_k[min(k, len(success_k)) - 1]:.3f}"
        print(row)


if __name__ == "__main__":
    main()