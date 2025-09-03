from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _sort_models_dict(
    models_dict: Dict[str, float],
) -> Tuple[List[Tuple[str, float]], Tuple[str, ...], Tuple[float, ...]]:
    """Sort a dictionary of models by their values.

    Args:
        models_dict: Dictionary mapping model names to their numeric values.

    Returns:
        A tuple containing:
            - sorted_models: List of (label, value) tuples sorted by value
            - labels: Tuple of sorted model names
            - values: Tuple of sorted model values
    """
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_models)
    return sorted_models, labels, values


def plot_line_comparison(
    models_dict: Dict[str, float],
    xlabel: str = "\nMinimal Description Length (MDL)",
    title: str = "Model Comparison\n",
    padding: int = 10_000,
    xrange: Optional[Tuple[float, float]] = None,
    print_decimals: bool = False,
    filename: Optional[str] = None,
) -> None:
    """Plot a line comparison chart showing model values as points on a line.

    Creates a horizontal line plot with model values as points, alternating
    labels above and below the line for readability. Useful for comparing
    model performance metrics like MDL scores.

    Args:
        models_dict: Dictionary mapping model names to their numeric values.
        xlabel: Label for the x-axis.
        title: Title for the plot.
        padding: Padding to add to x-axis limits when xrange is not specified.
        xrange: Tuple of (start, end) for x-axis range. If None, uses padding.
        print_decimals: Whether to format x-axis labels with decimal places.
        filename: If provided, saves the plot to this filename.

    Returns:
        None. Displays the plot and optionally saves to file.
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
