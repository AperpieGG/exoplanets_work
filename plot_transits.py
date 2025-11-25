#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import allesfitter
import seaborn as sns
import warnings
from plot_images import plot_images

warnings.filterwarnings("ignore")
plot_images()

TARGET = 'TIC-453'

# Set seaborn plot style
sns.set(context='paper', style='ticks', palette='deep',
        font='sans-serif', font_scale=2, color_codes=True)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

# --- Load dataset ---
print("Loading CCD dataset...")
alles = allesfitter.allesclass(f'data/allesfit_{TARGET}')

# --- Create figure: 2 rows, 1 column ---
fig, axes = plt.subplots(2, 1, figsize=(8, 6),
                         gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                         sharex=True, dpi=300)

# --- Plot TESS phasezoom in the first row ---
print("Plotting TESS phasezoom...")
alles.plot('sg1', 'b', 'phasezoom', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('')

# --- Plot TESS residuals in the second row ---
print("Plotting TESS residuals...")
alles.plot('sg1', 'b', 'phasezoom_residuals', ax=axes[1])
# remove ticks lavels from top plot

# --- Formatting ---
axes[0].set_ylabel('Relative flux')
axes[1].set_ylabel('Residuals')
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-0.003, 0.003)

# --- Show figure ---
# --- Save figure ---
path = f'data/allesfit_{TARGET}/'
# tight layout to avoid cutting off labels
fig.tight_layout()
fig.savefig(path + 'sg1_LC.pdf', bbox_inches='tight')
plt.show()
