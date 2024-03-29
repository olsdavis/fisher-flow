"""Useful functions for plotting (namely, style)."""
import matplotlib.pyplot as plt
import seaborn as sns


def define_style():
    """
    Sets the style up for matplotlib.
    """
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r"""\usepackage[T1]{fontenc}"""
    plt.rc("font", family="serif", weight="normal", size=16)
    sns.set_theme()
    sns.set_style(style="whitegrid")
    sns.set_palette("Paired")


def save_plot(loc: str):
    """
    Saves the current matplotlib plot at location `loc`. This is useful
    to keep the same export parameters for all the plots.
    """
    plt.savefig(
        loc, bbox_inches="tight", dpi=300,
    )
