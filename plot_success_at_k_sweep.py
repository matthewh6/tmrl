"""
Appendix Success@K sweep: all tcont values for CSP.
Best tcont auto-selected. Inline end-labels, no legend box.

Usage:
    python plot_success_at_k_sweep.py \
        --json  results.json \
        --title "cube-single" \
        --out   success_at_k_sweep.pdf
"""

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import shared_style
matplotlib.rcParams.update(shared_style.RC)

TCONT_KEYS   = ["tmrl_t0.00", "tmrl_t0.25", "tmrl_t0.50", "tmrl_t0.75", "tmrl_t1.00"]
TCONT_LABELS = ["0.00",       "0.25",       "0.50",       "0.75",       "1.00"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_curve(data, key):
    pak   = data[key]["success_at_k"]
    k_max = max(int(k) for k in pak)
    arr   = np.zeros(k_max)
    for k_str, v in pak.items():
        arr[int(k_str) - 1] = v
    return arr


def auto_best_tcont(data):
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


def place_labels(end_points, ymin, ymax, min_gap_frac=0.08):
    """Spread inline labels to avoid vertical overlap."""
    y_range = ymax - ymin
    min_gap = min_gap_frac * y_range
    pts     = sorted(end_points, key=lambda p: p[1], reverse=True)
    label_ys = [pts[0][1]]
    for i in range(1, len(pts)):
        y = pts[i][1]
        if label_ys[-1] - y < min_gap:
            y = label_ys[-1] - min_gap
        label_ys.append(y)
    if label_ys[-1] < ymin:
        shift    = ymin - label_ys[-1]
        label_ys = [y + shift for y in label_ys]
    return {pt[2]: ly for pt, ly in zip(pts, label_ys)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",        required=True)
    parser.add_argument("--best_tcont",  default=None)
    parser.add_argument("--title",       default=None)
    parser.add_argument("--out",         default="success_at_k_sweep.pdf")
    parser.add_argument("--figsize",     nargs=2, type=float,
                        default=list(shared_style.FIGSIZE_SINGLE))
    parser.add_argument("--dpi",         type=int, default=300)
    args = parser.parse_args()

    data       = load_json(args.json)
    best_tcont = args.best_tcont or auto_best_tcont(data)
    print(f"Best tcont: {best_tcont}")

    fig, ax      = plt.subplots(figsize=tuple(args.figsize))
    all_curves   = []
    end_points   = []
    k_max_global = 0

    for idx, (key, raw_label) in enumerate(zip(TCONT_KEYS, TCONT_LABELS)):
        if key not in data:
            continue

        success_k = extract_curve(data, key)
        k_max  = len(success_k)
        k_max_global = max(k_max_global, k_max)
        k_vals = np.arange(1, k_max + 1)

        color   = shared_style.CSP_SWEEP[idx]
        is_best = (key == best_tcont)
        lw      = 2.0 if is_best else 1.2
        alpha   = 1.0 if is_best else 0.75

        ax.plot(k_vals, success_k,
                color=color, linestyle="-",
                linewidth=lw, alpha=alpha,
                zorder=4 if is_best else 3)

        all_curves.append(success_k)
        label = f"$\\sigma={raw_label}$"
        if is_best:
            label += " (best)"
        end_points.append((k_max, float(success_k[-1]), label, color, is_best))

    ymin, ymax  = smart_ylim(all_curves)
    label_y_map = place_labels(end_points, ymin, ymax, min_gap_frac=0.08)

    x_curve_end = k_max_global
    x_label     = k_max_global + 0.6

    for (kmax, y_curve, label, color, is_best) in end_points:
        y_lbl = label_y_map[label]
        fw    = "semibold" if is_best else "normal"

        ax.annotate("",
                    xy=(x_label - 0.1, y_lbl),
                    xytext=(x_curve_end + 0.1, y_curve),
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.6, alpha=0.6),
                    xycoords="data", textcoords="data",
                    clip_on=False)

        ax.text(x_label, y_lbl, label,
                color=color, fontsize=7.5, va="center",
                fontweight=fw, clip_on=False,
                transform=ax.transData)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(clean_yticks(ymin, ymax))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(1, k_max_global + 4.0)
    ax.set_xticks([k for k in [1, 5, 10, 20] if k <= k_max_global])
    ax.set_xlabel("$K$")
    ax.set_ylabel("Success@$K$")

    title = args.title
    if title:
        title = shared_style.TASK_LABELS.get(title, title)
        ax.set_title(title, pad=4)

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved -> {args.out}")

    ks = [1, 5, 10, 20]
    print(f"\n{'tcont':<16}" + "".join(f"  Success@{k:>2}" for k in ks))
    print("-" * 46)
    for key, label, success_k in zip(TCONT_KEYS, TCONT_LABELS, all_curves):
        best_str = " (best)" if key == best_tcont else "       "
        row = f"s={label}{best_str}"
        for k in ks:
            row += f"  {success_k[min(k, len(success_k)) - 1]:.3f}"
        print(row)


if __name__ == "__main__":
    main()