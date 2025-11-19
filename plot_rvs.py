#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import
import seaborn as sns
import numpy as np
from plot_images import plot_images
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import allesfitter
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})
plot_images()
alles = allesfitter.allesclass('data/allesfit_TIC-453')
print(alles)

for style in ['phase']:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)

    alles.plot('HARPS', 'b', style, ax=axes[0], color='blue')
    alles.plot('CORALIE_1', 'b', style, ax=axes[0], color='green')
    alles.plot('CORALIE_2', 'b', style, ax=axes[0], color='orange')
    axes[0].set_title('')

    alles.plot('HARPS', 'b', style + '_residuals', ax=axes[1], color='blue')
    alles.plot('CORALIE_1', 'b', style + '_residuals', ax=axes[1], color='green')
    alles.plot('CORALIE_2', 'b', style + '_residuals', ax=axes[1], color='orange')
    axes[1].set_title('')
    path = 'data/allesfit_TIC-453/'
    # fig.savefig(path + 'TIC_RV.pdf', bbox_inches='tight')

plt.show()

for style in ['full_minus_offset']:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)

    instruments = ['HARPS', 'CORALIE_1', 'CORALIE_2']
    colors = ['blue', 'green', 'orange']

    # Plot data
    for inst, color in zip(instruments, colors):
        alles.plot(inst, 'b', style, ax=axes[0], color=color)

    axes[0].set_title('')

    # Residuals
    for inst, color in zip(instruments, colors):
        alles.plot(inst, 'b', 'full_residuals', ax=axes[1], color=color)

    # Horizontal line
    axes[1].axhline(0, color='indianred', linestyle='-', lw=1.2)
    axes[1].set_title('')

    # Shift x-axis by 2,400,000
    offset = 2457000
    axes[1].set_xlabel(f'Time (BJD - {offset})')
    axes[0].set_xlabel(f'Time (BJD - {offset})')  # optional if you want label on top panel too

    # Shift the tick labels
    for ax in axes:
        ticks = ax.get_xticks()
        ax.set_xticklabels([f'{int(t - offset)}' for t in ticks])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    path = 'data/allesfit_TIC-453/'
    fig.savefig(path + 'TIC_RV_TIME.pdf', bbox_inches='tight')

plt.show()


