"""Charts for the report (matplotlib PNGs). Every chart has labeled axes with units, and its
numbers are also in a table in the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order; validated for color-blind safety
PLAN_COLORS = {"Current": COLORS[0], "Webster": COLORS[1], "Optimized": COLORS[2]}
APPROACH_COLORS = dict(zip("NSEW", COLORS))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def grouped_bars(table: pd.DataFrame, category: str, series: str, value: str, colors: dict, title: str,
                 xlabel: str, ylabel: str, path: Path, errors: tuple[str, str] | None = None) -> Path:
    """Bars per category, one bar per series, value printed on each bar."""
    cats = list(dict.fromkeys(table[category]))
    names = [s for s in colors if s in set(table[series])] + sorted(set(table[series]) - set(colors))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, name in enumerate(names):
        rows = table[table[series] == name].set_index(category).reindex(cats)
        x = np.arange(len(cats)) + (i - (len(names) - 1) / 2) * width
        yerr = None
        if errors:
            yerr = [rows[value] - rows[errors[0]], rows[errors[1]] - rows[value]]
        bars = ax.bar(x, rows[value], width * 0.92, color=colors.get(name, MUTED), label=str(name),
                      yerr=yerr, ecolor=INK, capsize=3 if errors else 0, linewidth=0)
        tops = rows[errors[1]] if errors else rows[value]
        for bar, number, top in zip(bars, rows[value], tops):
            if pd.notna(number):
                ax.annotate(f"{number:.1f}" if number < 100 else f"{number:.0f}",
                            (bar.get_x() + bar.get_width() / 2, max(bar.get_height(), top)), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(cats)), [str(c) for c in cats])
    _style(ax, title, xlabel, ylabel)
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=9, ncol=len(names), loc="upper left", bbox_to_anchor=(0, -0.18))
    return _save(fig, path)


def model_scatter(table: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for approach in "NSEW":
        rows = table[table["approach"] == approach]
        if len(rows):
            ax.scatter(rows["measured_s"], rows["predicted_s"], s=36, color=APPROACH_COLORS[approach],
                       edgecolor="#fcfcfb", linewidth=1.5, label=f"from {approach}")
    top = float(max(table["measured_s"].max(), table["predicted_s"].max()) * 1.1) if len(table) else 1
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--", label="perfect match")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    _style(ax, title, "measured control delay (s per vehicle)", "model's predicted delay (s per vehicle)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def cycle_curve(curve: pd.DataFrame, chosen: float, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ok = curve.dropna()
    ax.plot(ok["cycle_s"], ok["veh_hours_per_hour"], color=COLORS[2], linewidth=2)
    best = ok.loc[(ok["cycle_s"] - chosen).abs().idxmin()] if len(ok) else None
    if best is not None:
        ax.scatter([best["cycle_s"]], [best["veh_hours_per_hour"]], s=64, color=COLORS[2], edgecolor="#fcfcfb",
                   linewidth=2, zorder=3)
        ax.annotate(f"recommended: {chosen:g} s", (best["cycle_s"], best["veh_hours_per_hour"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=INK)
    _style(ax, title, "cycle length (s)", "total delay (vehicle-hours per hour)")
    return _save(fig, path)
