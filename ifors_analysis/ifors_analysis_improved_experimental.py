#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS PROGRAM HAS NOT BEEN TESTED YET!!!
USE AT YOUR OWN RISK!!!


Temperature and Fitting Data Analysis Script
Created on 2025-01-21

Author: Philip Groß
This script processes temperature estimation data and fitting data for Raman spectroscopy analysis. 
It loads individual sample files, computes statistics, generates plots, and optionally 
aggregates results into summary files.


THIS PROGRAM HAS NOT BEEN TESTED YET!!!
USE AT YOUR OWN RISK!!!
"""

### Import packages ###
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import gridspec
import seaborn as sns

# Configure seaborn style
sns.set()
sns.set_style('ticks')

### Functions ###

def load_temperature_file_as_dataframe(temp_file_location):
    """Load temperature data into a DataFrame."""
    assert os.path.exists(temp_file_location), f"File not found: {temp_file_location}"
    with open(temp_file_location) as file:
        data = pd.read_csv(file, sep='\s+', header=0)
        data.columns = ['sample', 'analysis', 'temp']
    return data

def load_fit_file_as_dataframe(fit_file_location):
    """Load fitting data into a DataFrame."""
    assert os.path.exists(fit_file_location), f"File not found: {fit_file_location}"
    with open(fit_file_location) as file:
        df = pd.read_csv(file, sep='\s+', header=0, names=[
            'measurement_ID', 'D_STA', 'D_std', 'G_STA', 'G_std', 'G_shape_factor', 
            'Gsf_std', 'Dmax_pos', 'Dmax_std', 'Gmax_pos', 'Gmax_std', 
            'Dmax/Gmax-ratio', 'D/G_std'
        ])
    
    # Parse sample and analysis names
    dfid = pd.DataFrame(columns=['measurement_ID', 'sample', 'analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    dfid['sample'] = dfid.measurement_ID.str.split('_').str[0]
    dfid['analysis'] = dfid['measurement_ID'].str.split('_').str[1]
    
    # Combine parsed data with the main dataframe
    data = pd.concat([dfid, df], axis=1).drop(columns=['measurement_ID'])
    return data

def sample_statistics(data):
    """Calculate sample temperature statistics."""
    samplename = data.iloc[0]['sample']
    stats = data['temp'].describe()
    
    count = int(stats['count'])
    mean = int(round(stats['mean']))
    stdev = int(round(stats['std']))
    minval = int(round(stats['min']))
    maxval = int(round(stats['max']))
    median = int(round(stats['50%']))
    quart1 = int(round(stats['25%']))
    quart3 = int(round(stats['75%']))
    
    # Calculate 95% confidence interval
    lower_ci, upper_ci = st.t.interval(0.95, len(data['temp'])-1, loc=mean, scale=st.sem(data['temp']))
    lower_ci, upper_ci = int(round(lower_ci)), int(round(upper_ci))
    
    return samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci

def write_statistics_to_file(samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci):
    """Write sample statistics to a file."""
    if not os.path.exists('sample_statistics.txt'):
        with open('sample_statistics.txt', 'w') as f_out:
            f_out.write('#samplename count mean 95%CI stdev 25% median 75% min max\n')
    
    with open('sample_statistics.txt', 'a') as f_out:
        f_out.write(
            f"{samplename} {count} {mean} {upper_ci-mean} {stdev} {quart1} {median} {quart3} {minval} {maxval}\n"
        )

def plot_temperatures(data, samplename, mean, stdev, upper_ci, lower_ci, ax):
    """Plot temperature data for a sample."""
    alphavalue = 0.1
    ax = data['temp'].plot(use_index=True, color='black', marker='o', linewidth=1.5, label='_nolegend_')
    ax.set_title(f'{samplename} RSCM Temperature', fontsize=16)
    ax.set_ylabel('T (°C)', fontsize=14)
    ax.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    ax.axhspan(mean - stdev, mean + stdev, facecolor='0', alpha=alphavalue)
    ax.axhspan(lower_ci, upper_ci, facecolor='0', alpha=alphavalue)
    ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
    return ax.get_xlim(), ax.get_ylim()

def plot_kde(data, mean, stdev, upper_ci, lower_ci, ax, xlim, ylim):
    """Plot Kernel Density Estimate (KDE) for a sample."""
    alphavalue = 0.1
    sns.kdeplot(data['temp'], ax=ax, fill=False, legend=False, color='k', linewidth=1.5)
    ax.set_ylim(ylim)
    ax.set_xlim(left=0)
    ax.set_title('KDE', fontsize=16)
    ax.hlines(mean, 0, max(ax.get_xlim()), color='k', linestyle='dashed', linewidth=1)

def analyze_samples(temp_file_location, fit_file_location, write_results=False):
    """Analyze temperature and fitting data for all samples."""
    temp_aggregate = pd.DataFrame()
    fit_aggregate = pd.DataFrame()

    # Process individual temperature files
    for file in glob.glob(temp_file_location):
        temp_data = load_temperature_file_as_dataframe(file)
        temp_aggregate = pd.concat([temp_aggregate, temp_data])
        
        # Calculate and optionally save statistics
        stats = sample_statistics(temp_data)
        if write_results:
            write_statistics_to_file(*stats)
        
        # Plot data
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
        ax0 = plt.subplot(gs[0])
        xlim, ylim = plot_temperatures(temp_data, stats[0], stats[2], stats[3], stats[9], stats[8], ax0)
        ax1 = plt.subplot(gs[1])
        plot_kde(temp_data, stats[2], stats[3], stats[9], stats[8], ax1, xlim, ylim)
        plt.tight_layout()
        plt.show()
    
    # Process fitting data files
    for file in glob.glob(fit_file_location):
        fit_data = load_fit_file_as_dataframe(file)
        fit_aggregate = pd.concat([fit_aggregate, fit_data])
    
    # Merge data
    aggregate = pd.merge_ordered(fit_aggregate, temp_aggregate, on=['sample', 'analysis'])
    aggregate['T_STA'] = np.where(aggregate['G_shape_factor'] > 3, aggregate['G_STA'], aggregate['D_STA'])

    # Plot aggregated data
    sns.violinplot(data=aggregate.sort_values('temp'), x='sample', y='temp')
    plt.show()

### Main Script ###
if __name__ == "__main__":
    # Specify file paths
    temp_file_location = '/path/to/temperature_estimates/*.txt'
    fit_file_location = '/path/to/fitting_data/*.fit'
    analyze_samples(temp_file_location, fit_file_location, write_results=True)

