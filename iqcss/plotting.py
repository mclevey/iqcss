from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

def _sort_models_dict(models_dict):
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_models)
    return sorted_models, labels, values

def plot_line_comparison(
    models_dict,
    xlabel="\nMinimal Description Length (MDL)",
    title="Model Comparison\n",
    padding=10_000,
    xrange=None,
    print_decimals=False,
    filename=None,
):
    """
    xrange: tuple of start and end for horizontal line and xaxis (e.g., 0, 1)
    """
    sorted_models, labels, values = _sort_models_dict(models_dict)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    if xrange is not None:
        xstart, xend = xrange
    else:  # use padding
        xstart = min(values) - padding
        xend = max(values) + padding

    plt.hlines(1, xstart, xend, color="black", linestyles="solid")
    plt.xlim(xstart, xend)

    # alternate annotations above and below the line for readability
    for i, (label, value) in enumerate(sorted_models):
        ax.plot(value, 1, "o", label=label, c="black")
        if i % 2 == 0:
            ax.text(value, 1.02, f"{label}\n{value:,}", ha="center", va="bottom")
        else:
            ax.text(value, 0.98, f"{label}\n{value:,}", ha="center", va="top")

    plt.yticks([])
    plt.xlabel(xlabel)
    plt.title(title, loc="left", fontsize=14)

    if print_decimals is True:
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{round(x, 1):,}")
        )
    else:
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    if filename is not None:
        plt.savefig(filename, dpi=300)