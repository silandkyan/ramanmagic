#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:56:42 2022

@author: pgross
"""

### load packages ###
#import sys
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as st
#from scipy.stats import kde
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import matplotlib.transforms as transforms
#import matplotlib.cbook as cbook
from matplotlib import gridspec
#from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set()
sns.set_style('ticks')


### User-defined functions ###

def load_temperature_file_as_dataframe(temp_file_location):
    """
    Loads a temperature data file into a pandas DataFrame, processes the data, 
    and returns a cleaned DataFrame ready for analysis or plotting.

    Args:
        temp_file_location (str): The file path to the temperature data file.

    Returns:
        pd.DataFrame: A processed DataFrame with columns ['sample', 'temp'], 
                      indexed by 'analysis', and sorted by temperature.

    Raises:
        AssertionError: If the file does not exist at the specified location.

    Description:
    1. Reads the temperature file and loads it into a pandas DataFrame, 
       with column names: ['measurement_ID', 'temp', 'se', 'conf', 'pred'].
    2. Extracts metadata (sample and analysis) from 'measurement_ID' into a helper DataFrame.
    3. Combines the helper DataFrame with the original data, removing unnecessary columns 
       ('measurement_ID', 'se', 'conf', 'pred').
    4. Sorts the DataFrame by temperature in ascending order and sets the index to 'analysis'.
    """
    assert os.path.exists(temp_file_location), "I did not find the file at " + str(temp_file_location)
    
    # Load the file into a pandas DataFrame
    with open(temp_file_location) as file:
        df = pd.read_csv(file, names=['measurement_ID', 'temp', 'se', 'conf', 'pred'], 
                         sep='\s+', header=0)
    
    # Create a helper DataFrame to store 'sample' and 'analysis' extracted from 'measurement_ID'
    dfid = pd.DataFrame(columns=['measurement_ID', 'sample', 'analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0)
    dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(1)
    
    # Combine the helper DataFrame with the main DataFrame
    data = pd.concat([dfid, df], axis=1)
    
    # Drop unnecessary columns
    data = data.drop('measurement_ID', axis=1)
    data = data.drop('se', axis=1)
    data = data.drop('conf', axis=1)
    data = data.drop('pred', axis=1)
    
    # Sort by temperature and reset index to 'analysis'
    data = data.sort_values('temp')
    data = data.set_index('analysis')
    
    return data

def load_fit_file_as_dataframe(fit_file_location):
    """
    Loads a fit data file into a pandas DataFrame, processes the data, 
    and returns a cleaned DataFrame with metadata extracted.

    Args:
        fit_file_location (str): The file path to the fit data file.

    Returns:
        pd.DataFrame: A processed DataFrame with metadata columns ['sample', 'analysis'] 
                      combined with the fit data.

    Raises:
        AssertionError: If the file does not exist at the specified location.

    Description:
    1. Reads the fit data file into a pandas DataFrame, using specified column names:
       ['measurement_ID', 'D_STA', 'D_std', 'G_STA', 'G_std', 'G_shape_factor',
       'Gsf_std', 'Dmax_pos', 'Dmax_std', 'Gmax_pos', 'Gmax_std', 
       'Dmax/Gmax-ratio', 'D/G_std'].
    2. Extracts metadata (sample and analysis) from 'measurement_ID' into a helper DataFrame.
    3. Combines the metadata with the main DataFrame.
    4. Removes the 'measurement_ID' column from the final DataFrame.
    """
    # Verify that the file exists
    assert os.path.exists(fit_file_location), "I did not find the file at " + str(fit_file_location)
    
    # Load the file into a pandas DataFrame
    with open(fit_file_location) as file:
        df = pd.read_csv(file, names=['measurement_ID', 'D_STA', 'D_std', 
                                      'G_STA', 'G_std', 'G_shape_factor', 
                                      'Gsf_std', 'Dmax_pos', 'Dmax_std', 'Gmax_pos', 
                                      'Gmax_std', 'Dmax/Gmax-ratio', 'D/G_std'],
                         sep='\s+', header=0)
    
    # Create a helper DataFrame to extract 'sample' and 'analysis' from 'measurement_ID'
    dfid = pd.DataFrame(columns=['measurement_ID', 'sample', 'analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0)
    dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(1)
    
    # Combine metadata with the main DataFrame
    data = pd.concat([dfid, df], axis=1)
    
    # Drop the 'measurement_ID' column from the final DataFrame
    data = data.drop('measurement_ID', axis=1)
    
    return data

def sample_statistics(data):
    """
    Computes descriptive statistics and a 95% confidence interval (CI) for the mean 
    temperature (temp) column in the given dataset.

    Args:
        data (pd.DataFrame): A pandas DataFrame with at least two columns:
                             - 'sample': Identifies the sample name.
                             - 'temp': Contains the temperature values for analysis.

    Returns:
        tuple: A tuple containing the following statistics:
               (sample_name, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci)
               - sample_name (str): The name of the sample, taken from the first row.
               - count (int): The total number of temperature measurements.
               - mean (int): The mean temperature.
               - stdev (int): The standard deviation of the temperatures.
               - minval (int): The minimum temperature.
               - maxval (int): The maximum temperature.
               - median (int): The median temperature.
               - quart1 (int): The 1st quartile temperature (25th percentile).
               - quart3 (int): The 3rd quartile temperature (75th percentile).
               - lower_ci (int): The lower bound of the 95% confidence interval for the mean.
               - upper_ci (int): The upper bound of the 95% confidence interval for the mean.

    Description:
    1. Extracts the sample name from the 'sample' column.
    2. Computes key statistics for the 'temp' column:
       - count, mean, standard deviation, min, max, median, 1st quartile, 3rd quartile.
    3. Calculates the 95% confidence interval for the mean using Student's t-test.

    Note:
    - The confidence interval calculation uses scipy's `st.t.interval` and accounts for
      the sample size and standard error of the mean (st.sem).
    - All statistics are rounded to the nearest integer.
    """
    # Get sample name from the first row of the 'sample' column
    samplename = data.iloc[0]['sample']
    
    # Gather statistics for the 'temp' column
    stats = data['temp'].describe()
    count = int(round(stats[0]))  # Number of elements
    mean = int(round(stats[1]))   # Mean temperature
    stdev = int(round(stats[2]))  # Standard deviation of temperature
    minval = int(round(stats[3])) # Minimum temperature
    maxval = int(round(stats[7])) # Maximum temperature
    median = int(round(stats[5])) # Median temperature
    quart1 = int(round(stats[4])) # 1st quartile (25th percentile)
    quart3 = int(round(stats[6])) # 3rd quartile (75th percentile)
    
    # Calculate the 95% confidence interval for the mean
    lower_ci, upper_ci = st.t.interval(
        0.95, len(data['temp']) - 1, loc=mean, scale=st.sem(data['temp'])
    )
    lower_ci = int(np.round(lower_ci))  # Lower bound of CI
    upper_ci = int(np.round(upper_ci))  # Upper bound of CI
    
    return samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci

def write_statistics_to_file(samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci):
    """
    Writes sample statistics to a text file named 'sample_statistics.txt'. If the file does not 
    exist, it creates the file and writes a header line first.

    Args:
        samplename (str): The name of the sample.
        count (int): Number of temperature measurements.
        mean (int): Mean temperature.
        stdev (int): Standard deviation of the temperature.
        minval (int): Minimum temperature.
        maxval (int): Maximum temperature.
        median (int): Median temperature.
        quart1 (int): 1st quartile (25th percentile) temperature.
        quart3 (int): 3rd quartile (75th percentile) temperature.
        lower_ci (int): Lower bound of the 95% confidence interval for the mean.
        upper_ci (int): Upper bound of the 95% confidence interval for the mean.

    Description:
    1. Checks if the file 'sample_statistics.txt' exists:
       - If not, it creates the file and writes a header line: 
         '#samplename count mean 95%CI stdev 25% median 75% min max'.
    2. Appends the statistics for the given sample to the file in the following format:
       'samplename count mean 95%CI stdev 25% median 75% min max'.
       - The 95% CI is represented by the difference (upper_ci - mean).

    Notes:
    - The function appends new data without overwriting the existing content.
    - The file is stored in the current working directory.

    """
    # Check if the file exists; if not, create it and write the header line
    if not os.path.exists('sample_statistics.txt'):
        with open('sample_statistics.txt', 'a') as f_out:
            f_out.write('#samplename count mean 95%CI stdev 25% median 75% min max' + '\n')
    
    # Append the statistics for the given sample to the file
    with open('sample_statistics.txt', 'a') as f_out:
        f_out.write(
            samplename 
            + ' ' + str(count) 
            + ' ' + str(mean) 
            + ' ' + str(upper_ci - mean)  # 95% CI (upper bound - mean)
            + ' ' + str(stdev)
            + ' ' + str(quart1)
            + ' ' + str(median) 
            + ' ' + str(quart3)
            + ' ' + str(minval)
            + ' ' + str(maxval)
            + '\n'
        )

def plot_temperatures(data, samplename, mean, stdev, upper_ci, lower_ci, ax):
    """
    Creates a plot of RSCM temperatures for a given sample, including mean temperature,
    standard deviation, and confidence intervals.

    Args:
        data (pd.DataFrame): DataFrame containing temperature values in the 'temp' column.
        samplename (str): The name of the sample, used in the plot title.
        mean (float): Mean temperature of the sample.
        stdev (float): Standard deviation of the sample's temperature.
        upper_ci (float): Upper bound of the 95% confidence interval for the mean.
        lower_ci (float): Lower bound of the 95% confidence interval for the mean.
        ax (matplotlib.axes._axes.Axes): Matplotlib Axes object where the plot will be drawn.

    Returns:
        tuple: The x-axis and y-axis limits of the plot as `(xlim, ylim)`.

    Description:
    1. Plots the temperature data as points with a black line connecting them.
    2. Adds title and y-axis label for clarity.
    3. Configures x-axis:
       - Variant 1: Hides the x-axis labels and ticks.
       - Variant 2 (commented out): Displays x-axis labels for individual analyses.
    4. Draws the following annotations:
       - A dashed line for the mean temperature with a text label.
       - A shaded region for ±1 standard deviation with a text label.
       - A shaded region for the 95% confidence interval.
    5. Adjusts plot limits to improve visualization.
    
    Note:
    - The function uses Matplotlib's `transforms` to correctly position the labels for the mean
      and standard deviation.
    - By default, the x-axis is formatted without labels (Variant 1).

    """
    alphavalue = 0.1  # Transparency for shaded regions
    
    ### PLOT ###
    # Plot the temperature data as points with a connecting line
    ax = data['temp'].plot(use_index=True, color='black', marker='o', linewidth=1.5, label='_nolegend_')
    
    # Add plot title and y-axis label
    ax.set_title(samplename + ' RSCM temperature', fontsize=16)
    ax.set_ylabel('T (°C)', fontsize=14)
        
    # Adds space to the left and right of plotted markers
    ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
    
    # Variant 1: Hide x-axis labels and ticks
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel('')
    
    # Variant 2: Show individual analyses on x-axis (commented out)
    # ax.set_xlabel('individual analyses', fontsize=14)
    # plt.xticks(range(len(data.index)), data.index, fontsize=10, rotation='vertical')
    
    # Specify positional parameters for text annotations
    stdy, meany = -13 / 72., 5 / 72.  # Position offsets in inches
    xshift = 0.01  # Horizontal text offset in axes coordinates

    # Perform translations for correct annotation placement
    dx = 0 / 72.  # Additional x-offset (currently 0)
    stdoffset = transforms.ScaledTranslation(dx, stdy, fig.dpi_scale_trans)
    meanoffset = transforms.ScaledTranslation(dx, meany, fig.dpi_scale_trans)
    stdshift = ax.transData + stdoffset
    meanshift = ax.transData + meanoffset
    stdtrans = transforms.blended_transform_factory(ax.transAxes, stdshift)
    meantrans = transforms.blended_transform_factory(ax.transAxes, meanshift)

    # Add horizontal line at the mean temperature with a label
    ax.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    ax.text(
        xshift, mean, 
        'mean = ' + str(mean) + '$\pm$' + str(upper_ci - mean) + '°C', 
        transform=meantrans, fontsize=12
    )

    # Add horizontal shaded region for ±1 standard deviation with a label
    ax.axhspan((mean - stdev), (mean + stdev), facecolor='0', alpha=alphavalue)
    ax.text(
        xshift, mean + stdev, 
        '$1\sigma$ = $\pm$' + str(stdev) + '°C', 
        transform=stdtrans, fontsize=12
    )

    # Add horizontal shaded region for 95% confidence interval
    ax.axhspan(lower_ci, upper_ci, facecolor='0', alpha=alphavalue)
    
    # Return x-axis and y-axis limits
    return ax.get_xlim(), ax.get_ylim()

def plot_kde(data, mean, stdev, upper_ci, lower_ci, ax, xlim, ylim):
    """
    Creates a kernel density estimation (KDE) plot of temperature data and highlights
    specific regions, including the standard deviation range and 95% confidence interval.

    Args:
        data (pd.DataFrame): DataFrame containing the temperature data in the 'temp' column.
        mean (float): Mean temperature of the sample.
        stdev (float): Standard deviation of the sample's temperature.
        upper_ci (float): Upper bound of the 95% confidence interval for the mean.
        lower_ci (float): Lower bound of the 95% confidence interval for the mean.
        ax (matplotlib.axes._axes.Axes): Matplotlib Axes object where the KDE plot will be drawn.
        xlim (tuple): The x-axis limits for the plot.
        ylim (tuple): The y-axis limits for the plot.

    Description:
    1. **KDE Plot**:
       - Generates a kernel density estimation curve for the temperature data using Seaborn.
       - Uses a black line (`color='k'`) with a linewidth of 1.5.
    2. **Shaded Regions**:
       - Fills the area under the KDE curve for:
         a. ±1 standard deviation (`mean ± stdev`).
         b. 95% confidence interval (`lower_ci` to `upper_ci`).
    3. **Customization**:
       - Removes x- and y-axis labels, ticks, and formatting for a clean look.
       - Adds a horizontal dashed line at the mean temperature.
       - Sets the title to "KDE" and enforces the specified `xlim` and `ylim`.

    Returns:
        None

    Notes:
    - The `sns.kdeplot` function creates the KDE curve, and data points are extracted
      from the curve to mask specific regions for shading.
    - Shading is achieved using `fill_betweenx`, with transparency controlled by `alphavalue`.

    """
    alphavalue = 0.1  # Transparency for shaded regions

    ### PLOT ###
    # Plot the KDE curve for the 'temp' column
    sns.kdeplot(
        data, y='temp', ax=ax, fill=False, legend=False, 
        color='k', linewidth=1.5
    )
    
    # Retrieve the last plotted line data (KDE curve)
    line = ax.get_lines()[-1]
    x, y = line.get_data()
    y = np.round(y, 0)
    
    # Mask data within the standard deviation range and fill the area
    mask = ((y <= mean + stdev) & (y >= mean - stdev))
    x, y = x[mask], y[mask]
    ax.fill_betweenx(y, x1=x, facecolor='k', alpha=alphavalue)

    # Mask data within the 95% confidence interval range and fill the area
    u, v = line.get_data()
    v = np.round(v, 0)
    mask_ci = ((v <= upper_ci) & (v >= lower_ci))
    u, v = u[mask_ci], v[mask_ci]
    ax.fill_betweenx(v, x1=u, facecolor='k', alpha=alphavalue)
    
    # Remove x- and y-axis ticks and labels for a clean appearance
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_ticks_position('none')

    # Set axis limits
    ax.set_ylim(ylim)
    ax.set_xlim(left=0)

    # Add title and remove axis labels
    ax.set_title('KDE', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add a horizontal dashed line at the mean temperature below the KDE curve
    ax.hlines(mean, 0, x[np.abs(y - mean).argmin()], color='k', linestyle='dashed', linewidth=1)

def plot_boxplot(data, ax):
    """
    Creates a boxplot for the temperature data to visualize its distribution,
    including outliers, without displaying ticks or labels.

    Args:
        data (pd.DataFrame): DataFrame containing the temperature data in the 'temp' column.
        ax (matplotlib.axes._axes.Axes): Matplotlib Axes object where the boxplot will be drawn.

    Description:
    1. **Boxplot Creation**:
       - Uses Seaborn's `sns.boxplot` to create a vertical boxplot.
       - Customizes the flier (outlier) properties using a black circular marker.
       - Sets the boxplot color scheme to black-and-white for simplicity.
    2. **Customization**:
       - Removes x- and y-axis ticks and labels for a clean appearance.
       - Adds a title ("boxplot") to the plot for clarity.
    
    Returns:
        None

    Notes:
    - `flierprops` defines the appearance of the outliers (markers).
    - The axis and tick labels are intentionally removed to make the boxplot minimalist.
    """

    # Customize the appearance of outliers
    flierprops = dict(marker='o', markerfacecolor='black', markersize=5, linestyle='none')

    ### PLOT ###
    # Create the boxplot with custom settings
    ax = sns.boxplot(
        data['temp'], color='white', width=0.4, flierprops=flierprops, 
        orient="v", linewidth=1.4
    )

    # Set the boxplot colors to black-and-white
    plt.setp(ax.artists, edgecolor='k', facecolor='w')  # Box edge and face colors
    plt.setp(ax.lines, color='k')  # Line colors (e.g., whiskers, medians)

    # Remove x- and y-axis ticks
    ax.set_xticks([]), ax.set_yticks([])

    # Remove y-axis labels and tick labels
    ax.set_ylabel(''), ax.set_yticklabels('')

    # Set the title of the plot
    ax.set_title('boxplot', fontsize=16)


### Analyze temperature and fitting data ###
'''
THIS PROGRAM HAS NOT BEEN TESTED YET!!!
USE AT YOUR OWN RISK!!!


This script processes temperature estimation data and fitting data for Raman spectroscopy analysis. 
It loads individual sample files, computes statistics, generates plots for each sample, and optionally 
aggregates results into summary files. Finally, it plots violin plots for all aggregated sample data.

Libraries used:
- pandas (for data manipulation)
- matplotlib (for plotting)
- seaborn (for KDE and violin plots)
- numpy, scipy (for statistical calculations)
- glob (for file path handling)


THIS PROGRAM HAS NOT BEEN TESTED YET!!!
USE AT YOUR OWN RISK!!!
'''

# File paths for temperature and fitting data
temp_file_location = '/media/pgross/gross-data/analytical_data/raman/graphitization/results/MO14/temperature_estimates.txt'
fit_file_location = '/home/pgross/heiBOX/programming/ifors_analysis/testdata/PG*_data.fit'

# Flag to decide whether to write statistical results to a file
write_results_to_file = False

# DataFrames to store aggregated data
temp_aggregate = pd.DataFrame()
fit_aggregate = pd.DataFrame()

### Analyze individual sample temperature data ###
for file in list(glob.glob(temp_file_location)):
    # Load temperature data into a DataFrame
    temp_data = load_temperature_file_as_dataframe(file)
    temp_aggregate = pd.concat([temp_aggregate, temp_data])
    
    # Calculate statistics for the current sample
    samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci = sample_statistics(temp_data)
    
    # Write sample statistics to a file if enabled
    if write_results_to_file:
        write_statistics_to_file(samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci)
    
    ### Plot individual sample data ###
    # Create a figure with subplots for temperature, KDE, and optionally boxplots
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 1])
    
    # Plot temperature data
    ax0 = plt.subplot(gs[0])
    xlim, ylim = plot_temperatures(temp_data, samplename, mean, stdev, upper_ci, lower_ci, ax0)
    
    # Plot KDE (Kernel Density Estimate)
    ax1 = plt.subplot(gs[1])
    plot_kde(temp_data, mean, stdev, upper_ci, lower_ci, ax1, xlim, ylim)
    
    # Optional: Uncomment to plot a boxplot
    # ax2 = plt.subplot(gs[2])
    # plot_boxplot(temp_data, ax2)
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    # Optional: Save plots to files
    # fig.savefig(samplename + '_result.png', format='png', bbox_inches='tight', dpi=300)
    # fig.savefig(samplename + '_result.pdf', format='pdf', bbox_inches='tight')


### Analyze and aggregate fitting data ###
for file in list(glob.glob(fit_file_location)):
    # Load fitting data into a DataFrame
    fit_data = load_fit_file_as_dataframe(file)
    fit_aggregate = pd.concat([fit_aggregate, fit_data])

# Merge temperature and fitting data, calculate additional parameters
aggregate = pd.merge_ordered(fit_aggregate, temp_aggregate, on=['sample', 'analysis'])
# Conditional calculation based on G_shape_factor threshold
aggregate['T_STA'] = np.where(aggregate['G_shape_factor'] > 3, aggregate['G_STA'], aggregate['D_STA'])


### Plot aggregated data ###
# Violin plot sorted by ascending temperature
sns.violinplot(aggregate.sort_values('temp'), x='sample', y='temp')

# Optional: Uncomment for boxplot
# sns.boxplot(aggregate.sort_values('temp'), x='sample', y='temp')

# Note: Calibration function for STA (from Lünsdorf et al., 2017) can be added here:
# STA = 42.879  # Example variable
# T = -8.259 * 1E-5 * STA**3 + 3.733 * 1E-2 * STA**2 - 6.445 * STA + 6.946 * 1E2

