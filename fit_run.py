#!/usr/bin/env python

import allesfitter
import argparse
import os
from allesfitter import allesclass
import sys
from allesfitter.v0.generative_models import inject_lc_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle


# Default Path to Allesfitter Directory
DEFAULT_PATH = os.getcwd()

# Create Argument Parser
parser = argparse.ArgumentParser(
    prog='allesfitter',
    description='Run Allesfitter fits for a specified target.'
)
parser.add_argument('--target', type=str, required=True, help='Target star directory name')
parser.add_argument('--root', type=str, default=DEFAULT_PATH, help='Root path to target directory')
parser.add_argument('--allesdir', type=str, default=DEFAULT_PATH, help='Name of Allesfitter directory')
parser.add_argument('--fitmethod', type=str, default='ns',
                    choices=['ns', 'mcmc'], help='Sampling type: ns (nested sampling) or mcmc')

args = parser.parse_args()

# Construct Full Path
target_path = os.path.join(args.root, args.target, args.allesdir)
print(f'Running Allesfitter for {args.target} in {target_path}')

# Check if Path Exists
if not os.path.exists(target_path):
    sys.exit(f"Error: Target path '{target_path}' does not exist.")

# # Run Allesfitter Based on Fit Method
if args.fitmethod == 'ns':
    print(f"Running Nested Sampling for {args.target}...")
    allesfitter.show_initial_guess(target_path)
    allesfitter.ns_fit(target_path)
    allesfitter.ns_output(target_path)

elif args.fitmethod == 'mcmc':
    print(f"Running MCMC for {args.target}...")
    allesfitter.show_initial_guess(target_path)
    allesfitter.mcmc_fit(target_path)
    allesfitter.mcmc_output(target_path)

