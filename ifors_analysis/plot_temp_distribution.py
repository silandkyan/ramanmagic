#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script takes temperature estimates from IFORS, calculates mean
temperature, standard deviation and confidence interval of a sample and 
plots them nicely. 

written by: Philip Groß, 2019, GEO FU Berlin
philip.gross@fu-berlin.de
"""

# load packages:
#import sys
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import matplotlib.transforms as transforms
#import matplotlib.cbook as cbook
from matplotlib import gridspec
import seaborn as sns
sns.set()
sns.set_style('ticks')

# load own functions:
#from read_file import read_file
#from matplotlib.ticker import FormatStrFormatter

print('Choose a file that contains peak-fitting data from ifors (_ifors_peaks.fit).')
print('You can also specify several files using wildcards!')
file_location = input("Enter the path of the file to process: ")
write_results = input("Do you want the summary of results to be written to a file? Type y/n; default = n --> ")
write_results_to_file = False
if write_results == 'y':
    write_results_to_file = True

for file in list(glob.glob(file_location)):
    #print(file_location)
    print(file)
    
    # check if file exists
    assert os.path.exists(file), "I did not find the file at "+str(file_location)
    f = open(file,'r+')
    filename = os.path.basename(file)
    f.close()

    # Load file as pd.table:
    df = pd.read_table(file, names=['measurement_ID','temp','se','conf','pred'], sep='\s+', header=0)
    
    # create new dataframe 'ident'
    dfid = pd.DataFrame(columns=['measurement_ID','sample','analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    
    # parse for sample and analysis names and write to new dataframe
    # original:
    dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0)
    dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(1)

    # for Al... samples:
    #dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0) + '_' +  dfid.measurement_ID.str.split('_').str.get(1)
    #dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(2)
    
    # combine dataframes to one and delete measurement_ID column
    data = pd.concat([dfid,df], axis=1)
    data = data.drop('measurement_ID', 1)
    
    
    ### gather statistics
    # sort dataframe by ascending T for plotting and reset index
    data_sorted = data.sort_values('temp')
    data_sorted = data_sorted.set_index('analysis')
    
    # get sample name string from first row of sample column
    samplename = data.iloc[0]['sample']
    
    # summarize statistics of dataset
    stats = data_sorted['temp'].describe()
    count = int(round(stats[0])) # number of elements
    mean = int(round(stats[1])) # calculate mean T
    stdev = int(round(stats[2])) # calculate stdev from T
    minval = int(round(stats[3]))
    maxval = int(round(stats[7]))
    median = int(round(stats[5])) # calculate median from T
    quart1 = int(round(stats[4]))
    quart3 = int(round(stats[6]))
    
    # calculate 95% confidence interval (CI) for the mean with student's t-test:
    lower_ci, upper_ci = st.t.interval(0.95, len(data_sorted['temp'])-1, loc=mean, scale=st.sem(data_sorted['temp']))
    
    lower_ci = int(np.round(lower_ci)) # lower bound of CI
    upper_ci = int(np.round(upper_ci)) # upper bound of CI
    
    # write statistics to file 
    if write_results_to_file == True:    
        with open('results.txt', 'a') as f_out:
            f_out.write(samplename 
                     +' '+ str(count) 
                     +' '+ str(mean) 
                     +' '+ str(upper_ci-mean) 
                     +' '+ str(stdev)
                     +' '+ str(quart1)
                     +' '+ str(median) 
                     +' '+ str(quart3)
                     +' '+ str(minval)
                     +' '+ str(maxval)
                     +' '+ '\n')
        
    ### plot individual T-data 
    # Create a figure of given size
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    alphavalue = 0.1
    
    ### Add a subplot
    ax0 = plt.subplot(gs[0])
    
    ax0 = data_sorted['temp'].plot(use_index=True, color='black', marker='s', linewidth=1.5)
    ax0.set_title(samplename + ' RSCM temperature')
    ax0.set_xlabel('')
    ax0.set_ylabel('T (°C)')
    
    # adds space left and right of the plotted markers
    ax0.set_xlim(ax0.get_xlim()[0] - 0.5, ax0.get_xlim()[1] + 0.5)
    
    # each index will be plotted
    plt.xticks(range(len(data_sorted.index)), data_sorted.index, fontsize=10, rotation='vertical')
    
    # automatically rotate x-axis caption to fit neatly
    #fig.autofmt_xdate()
    
    # specify positional parameters for text annotations
    stdy, meany = -13/72., 5/72. # Inches
    xshift = 0.01 # Axes coordinates
    
    # perform translations for correct annotation placement
    dx = 0/72.
    stdoffset = transforms.ScaledTranslation(dx, stdy, fig.dpi_scale_trans)
    meanoffset = transforms.ScaledTranslation(dx, meany, fig.dpi_scale_trans)
    stdshift = ax0.transData + stdoffset
    meanshift = ax0.transData + meanoffset
    stdtrans = transforms.blended_transform_factory(ax0.transAxes, stdshift)
    meantrans = transforms.blended_transform_factory(ax0.transAxes, meanshift)
    
    # add horizontal line at mean T+offset, with label
    ax0.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    ax0.text(xshift, mean, 'mean = ' + str(mean) + '$\pm$' + str(upper_ci-mean) + '°C', transform=meantrans)
    
    # add horizontal field at std+/-offset, with label
    ax0.axhspan((mean-stdev),(mean+stdev), facecolor='0', alpha=alphavalue)
    ax0.text(xshift, mean+stdev, '$1\sigma$ = $\pm$' + str(stdev) + '°C', transform=stdtrans)
    
    # add horizontal field for 95% CI
    ax0.axhspan(lower_ci, upper_ci, facecolor='0', alpha=alphavalue)
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    
    ### Add a subplot for KDE plot
    ax1 = plt.subplot(gs[1])
    #sns.kdeplot(data_sorted['temp'], ax=ax1, shade=True, 
    #            vertical=True, legend=False, color='k', linewidth=1.5, alpha=alphavalue)
    sns.kdeplot(data_sorted['temp'], ax=ax1, shade=False, 
                vertical=True, legend=False, color='k', linewidth=1.5)
    
    line = ax1.get_lines()[-1]
    x, y = line.get_data()
    y = np.round(y, 0)
    mask = ((y <= mean+stdev) & (y >= mean-stdev))
    x, y = x[mask], y[mask]
    ax1.fill_betweenx(y, x1=x, facecolor='k', alpha=alphavalue)
    
    u, v = line.get_data()
    v = np.round(v, 0)
    mask_ci = ((v <= upper_ci) & (v >= lower_ci))
    u, v = u[mask_ci], v[mask_ci]
    ax1.fill_betweenx(v, x1=u, facecolor='k', alpha=alphavalue)
    
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_ticks_position('none')
    ax1.set_ylim(ylim)
    ax1.set_title('KDE')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    ax1.hlines(mean, 0, x[np.abs(y-mean).argmin()], color='k', linestyle='dashed', linewidth=1)
    
    plt.tight_layout()
    #plt.show()
    
    ### export figure as .png
    fig.savefig(samplename +'_result')#, transparent=True)
