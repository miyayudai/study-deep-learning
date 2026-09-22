"""
Plotting utilities and publication-quality style setup.
"""
import os
import matplotlib.pyplot as plt

def setup_style():
    """Setup clean, modern, and publication-ready matplotlib style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.autolayout': True,
    })

def save_plot(fig_or_plt, filepath, dpi=300):
    """Save matplotlib figure to specified path with high DPI."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if hasattr(fig_or_plt, 'savefig'):
        fig_or_plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved successfully to: {filepath}")
