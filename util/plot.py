"""Useful functions for plotting (namely, style)."""
import matplotlib.pyplot as plt
import seaborn as sns


def define_style():
    """
    Sets the style up for matplotlib.
    """
    sns.set_theme(context="paper", style="whitegrid")
    sns.set_style()
