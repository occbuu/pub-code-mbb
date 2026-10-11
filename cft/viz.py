"""Figure styling for the manuscript.

Publication figures are static, print-first and light-mode only, so the design
system is reduced to its parameters: a validated four-slot categorical palette
for the four measurement arms, a single-hue sequential ramp for magnitude
(reliability, agreement heatmaps) and a blue<->red diverging pair with a neutral
gray midpoint for signed quantities (bias, attenuation).

The four method hues were validated with the palette checker on the *all-pairs*
list (scatter and small-multiple forms are used): worst CVD Delta E 9.2, worst
normal-vision Delta E 16.3, both above their gates.  The aqua slot sits below 3:1
against the paper surface, so every figure that uses it also carries direct
labels and a companion table -- the documented relief rule.
"""

from __future__ import annotations

import matplotlib as mpl
try:
    mpl.use("Agg")          # headless batch / nohup — no display needed
except Exception:
    pass
import matplotlib.pyplot as plt

# --- parameters -------------------------------------------------------------
METHOD_COLORS = {
    "DICT": "#2a78d6",     # blue      slot 1
    "TOPIC": "#eb6834",    # orange    slot 2
    "ENCODER": "#1baf7a",  # aqua      slot 3  (contrast relief: direct labels)
    "LLM": "#4a3aa7",      # violet    slot 7
    "LLM2": "#a13d6b",     # wine      second LLM (family robustness)
}
METHOD_ORDER = ["DICT", "TOPIC", "ENCODER", "LLM"]
METHOD_LABELS = {
    "DICT": "Expert dictionary",
    "TOPIC": "Topic model",
    "ENCODER": "Fine-tuned encoder",
    "LLM": "LLM (Qwen)",
    "LLM2": "LLM (Mistral)",
}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")

INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781",
           grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")


def sequential_cmap(name="cft_blue"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ_BLUE)


def diverging_cmap(name="cft_div"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(DIVERGING))


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=3, width=0.8)


def save(fig, path, also_pdf=True):
    fig.savefig(path, dpi=400)
    if also_pdf and str(path).endswith(".png"):
        fig.savefig(str(path)[:-4] + ".pdf")
    plt.close(fig)
    return path
