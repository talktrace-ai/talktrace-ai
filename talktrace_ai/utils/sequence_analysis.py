"""Code-transition analysis — what code follows which.

Frequency plots tell you *which* codes occur; a transition matrix tells
you the *order*. In a Mercer-style classroom-talk dataset, that's the
difference between "the teacher asks lots of explanation questions" and
"explanation questions reliably trigger elaborated student answers,
which the teacher then confirms with feedback." Same code counts, very
different conversational dynamics.

The matrix is built over consecutive coded turns. Uncoded turns are
skipped (they break the flow but aren't themselves transitions). With
multi-coding ("RE; A; CO" cells), only the first code is taken — the
priority resolution upstream already put the highest-priority code
first, so this preserves the dominant-code reading without exploding
the matrix into noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_code(cell) -> str:
    """Return the first non-empty token from a possibly multi-coded cell."""
    if cell is None:
        return ""
    s = str(cell).strip()
    if not s:
        return ""
    # Multi-coding cells use "; " as separator (see make_qualitative_stats_df).
    return s.split(";")[0].strip()


def build_transition_matrix(impulse_df, code_col: str, *, normalize: bool = True):
    """Return (codes, matrix_df, n_transitions) for an impulse DataFrame.

    Parameters
    ----------
    impulse_df : pd.DataFrame
        Rows in chronological order, with at least ``code_col`` (Shortcode).
    code_col : str
        Column name holding the per-turn code(s).
    normalize : bool, default True
        If True, each row sums to 1 (conditional probabilities P(j|i)).
        Rows whose source code never preceded another stay all-zero.

    Returns
    -------
    codes : list[str]
        Sorted unique codes that appear in any from/to position.
    matrix_df : pd.DataFrame
        Square DataFrame indexed and columned by ``codes``. Empty when
        ``codes`` is empty.
    n_transitions : int
        Count of consecutive (coded → coded) pairs. ``len(codes) ** 2 ==
        matrix_df.size`` and ``n_transitions == raw_counts.sum()``.
    """
    if impulse_df is None or impulse_df.empty or code_col not in impulse_df.columns:
        return [], pd.DataFrame(), 0

    # Pick the first code per turn; skip uncoded.
    seq = [_first_code(c) for c in impulse_df[code_col].tolist()]
    seq = [c for c in seq if c]
    if len(seq) < 2:
        codes = sorted(set(seq))
        empty = pd.DataFrame(0.0, index=codes, columns=codes) if codes else pd.DataFrame()
        return codes, empty, 0

    codes = sorted(set(seq))
    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    counts = np.zeros((n, n), dtype=float)
    n_transitions = 0
    for a, b in zip(seq[:-1], seq[1:]):
        counts[idx[a], idx[b]] += 1
        n_transitions += 1

    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid div-by-zero: rows with no outgoing transitions stay zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.where(row_sums > 0, counts / row_sums, 0.0)
    else:
        mat = counts

    return codes, pd.DataFrame(mat, index=codes, columns=codes), n_transitions


def plot_transition_heatmap(matrix_df, ax, *, title: str = "", value_fmt: str = "{:.0%}",
                            cmap_name: str = "Blues", text_color_threshold: float = 0.5):
    """Render a transition-matrix heatmap on the given matplotlib Axes.

    Cells are annotated with the matrix value (default: percentage).
    Cell text flips between dark/light depending on background lightness
    so the numbers stay readable against deep colours. Caller controls
    figure size and title styling — this only paints the matrix.
    """
    import matplotlib.pyplot as plt
    if matrix_df is None or matrix_df.empty:
        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax

    codes = list(matrix_df.index)
    data = matrix_df.values.astype(float)
    vmax = float(np.nanmax(data)) if data.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(data, cmap=cmap_name, aspect="equal", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(codes)))
    ax.set_yticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=0, fontsize=8)
    ax.set_yticklabels(codes, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    for i, _row in enumerate(codes):
        for j, _col in enumerate(codes):
            v = data[i, j]
            if v <= 0:
                continue
            txt = value_fmt.format(v)
            color = "white" if (v / vmax) >= text_color_threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_xlabel("→")
    ax.tick_params(top=False, bottom=True, left=True, right=False)
    return ax
