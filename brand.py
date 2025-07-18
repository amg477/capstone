import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

def apply_penta_style():
    """
    Apply Penta Group brand styling to matplotlib/seaborn plots.
    Sets fonts, colors, and layout defaults to match brand tone.
    """
    # Primary brand colors
    PRIMARY_GREEN = "#12715D"   # Deep green background
    ACCENT_GREEN  = "#4AB48E"   # Mint green accent

    # Font preferences (fallbacks if Penta custom fonts aren't available)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Georgia', 'Times New Roman', 'serif']
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12

    # Plot colors and grid
    sns.set_style("whitegrid")
    sns.set_palette([PRIMARY_GREEN, ACCENT_GREEN, "#E5F4F1", "#C8EADF"])

    # Apply layout settings
    rcParams['axes.edgecolor'] = PRIMARY_GREEN
    rcParams['axes.labelcolor'] = PRIMARY_GREEN
    rcParams['xtick.color'] = PRIMARY_GREEN
    rcParams['ytick.color'] = PRIMARY_GREEN
    rcParams['text.color'] = PRIMARY_GREEN
    rcParams['grid.color'] = "#E0E0E0"
    rcParams['axes.titleweight'] = 'bold'

    # Set figure facecolor
    rcParams['figure.facecolor'] = "white"
    rcParams['axes.facecolor'] = "white"