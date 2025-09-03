import importlib.resources as resources
from typing import Literal

import matplotlib.pyplot as plt


def set_style(
    style: Literal["statistical_rethinking"] = "statistical_rethinking",
) -> None:
    """Set matplotlib plotting style.

    Args:
        style: Style name to apply. Currently supports "statistical_rethinking"
               which uses McElreath style, or defaults to "fivethirtyeight".

    Returns:
        None. Sets the matplotlib style globally.
    """
    if style == "statistical_rethinking":
        style_path = resources.files("iqcss.mplstyles") / "mcelreath.mplstyle"
        plt.style.use(str(style_path))
    else:
        plt.style.use("fivethirtyeight")
