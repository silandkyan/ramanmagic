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
    assert os.path.exists(temp_file_location), "I did not find the file at "+str(temp_file_location)
    with open(temp_file_location) as file:
        df = pd.read_csv(file, names=['measurement_ID','temp','se','conf','pred'], 
                  sep='\s+', header=0)
    
    # create new helper dataframe 'ident':
    dfid = pd.DataFrame(columns=['measurement_ID','sample','analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    # parse for sample and analysis names and write to new dataframe:
    dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0)
    dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(1)
    # combine dataframes to one:
    data = pd.concat([dfid,df], axis=1)
    # delete unnecessary columns:
    data = data.drop('measurement_ID',1)
    data = data.drop('se',1)
    data = data.drop('conf',1)
    data = data.drop('pred',1)
    # sort dataframe by ascending T for plotting and reset index
    data = data.sort_values('temp')
    data = data.set_index('analysis')
    return data

def load_fit_file_as_dataframe(fit_file_location):
    assert os.path.exists(fit_file_location), "I did not find the file at "+str(fit_file_location)
    with open(fit_file_location) as file:
        df = pd.read_csv(file, names=['measurement_ID', 'D_STA', 'D_std', 
                                      'G_STA', 'G_std', 'G_shape_factor', 
                                      'Gsf_std', 'Dmax_pos', 'Dmax_std', 'Gmax_pos', 
                                      'Gmax_std', 'Dmax/Gmax-ratio', 'D/G_std'],
                         sep='\s+', header=0)
    
    # create new helper dataframe 'ident':
    dfid = pd.DataFrame(columns=['measurement_ID','sample','analysis'])
    dfid['measurement_ID'] = df.measurement_ID.str[4:]
    # parse for sample and analysis names and write to new dataframe:
    dfid['sample'] = dfid.measurement_ID.str.split('_').str.get(0)
    dfid['analysis'] = dfid.measurement_ID.str.split('_').str.get(1)
    # combine dataframes to one:
    data = pd.concat([dfid,df], axis=1)
    # delete unnecessary columns:
    data = data.drop('measurement_ID',1)
    return data

def sample_statistics(data):
    # get sample name string from first row of sample column
    samplename = data.iloc[0]['sample']
    # gather statistics
    stats = data['temp'].describe()
    count = int(round(stats[0])) # number of elements
    mean = int(round(stats[1])) # calculate mean T
    stdev = int(round(stats[2])) # calculate stdev from T
    minval = int(round(stats[3]))
    maxval = int(round(stats[7]))
    median = int(round(stats[5])) # calculate median from T
    quart1 = int(round(stats[4])) # 1st quartile
    quart3 = int(round(stats[6])) # 3rd quartile
    # calculate 95% confidence interval (CI) for the mean with student's t-test:
    # taken from Ulrich Stern's answer at:
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    lower_ci, upper_ci = st.t.interval(0.95, len(data['temp'])-1, loc=mean, scale=st.sem(data['temp']))
    lower_ci = int(np.round(lower_ci)) # lower bound of CI
    upper_ci = int(np.round(upper_ci)) # upper bound of CI
    #print(round(lower_ci), upper_ci)
    return samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci

def write_statistics_to_file(samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci):
    if os.path.exists('sample_statistics.txt') == False:
        with open('sample_statistics.txt', 'a') as f_out:
            f_out.write('#samplename count mean 95%CI stdev 25% median 75% min max' + '\n')
    with open('sample_statistics.txt', 'a') as f_out:
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

def plot_temperatures(data, samplename, mean, stdev, upper_ci, lower_ci, ax):
    alphavalue = 0.1
    ### PLOT ###
    ax = data['temp'].plot(use_index=True, color='black', marker='o', linewidth=1.5, label='_nolegend_')
    
    # title and ylabel
    ax.set_title(samplename + ' RSCM temperature', fontsize=16)
    ax.set_ylabel('T (°C)', fontsize=14)
        
    # adds space left and right of the plotted markers
    ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
    
    # Variant 1: no index will be plotted
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel('')
    
    # Variant 2: each index will be plotted
    #ax.set_xlabel('individual analyses', fontsize=14)
    #plt.xticks(range(len(data.index)), data.index, fontsize=10, rotation='vertical')
    
    # automatically rotate x-axis caption to fit neatly
    #fig.autofmt_xdate()
    
    # specify positional parameters for text annotations
    stdy, meany = -13/72., 5/72. # Inches
    xshift = 0.01 # in axes coordinates

    # perform translations for correct annotation placement
    dx = 0/72.
    stdoffset = transforms.ScaledTranslation(dx, stdy, fig.dpi_scale_trans)
    meanoffset = transforms.ScaledTranslation(dx, meany, fig.dpi_scale_trans)
    stdshift = ax0.transData + stdoffset
    meanshift = ax0.transData + meanoffset
    stdtrans = transforms.blended_transform_factory(ax0.transAxes, stdshift)
    meantrans = transforms.blended_transform_factory(ax0.transAxes, meanshift)

    # add horizontal line at mean T+offset, with label
    ax.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    ax.text(xshift, mean, 'mean = ' + str(mean) + '$\pm$' + str(upper_ci-mean) + '°C', transform=meantrans, fontsize=12)
    # add horizontal field at std+/-offset, with label
    ax.axhspan((mean-stdev),(mean+stdev), facecolor='0', alpha=alphavalue)
    ax.text(xshift, mean+stdev, '$1\sigma$ = $\pm$' + str(stdev) + '°C', transform=stdtrans, fontsize=12)
    # add horizontal field for 95% CI
    ax.axhspan(lower_ci, upper_ci, facecolor='0', alpha=alphavalue)
    
    return ax.get_xlim(), ax.get_ylim()

    
def plot_kde(data, mean, stdev, upper_ci, lower_ci, ax, xlim, ylim):
    alphavalue = 0.1
    ### PLOT ###
    sns.kdeplot(data, y='temp', ax=ax, fill=False, legend=False, 
                color='k', linewidth=1.5)
    
    # fill stdev-range below curve
    line = ax.get_lines()[-1]
    x, y = line.get_data()
    y = np.round(y, 0)
    mask = ((y <= mean+stdev) & (y >= mean-stdev))
    x, y = x[mask], y[mask]
    ax.fill_betweenx(y, x1=x, facecolor='k', alpha=alphavalue)

    # fill 95%ci-range below curve
    u, v = line.get_data()
    v = np.round(v, 0)
    mask_ci = ((v <= upper_ci) & (v >= lower_ci))
    u, v = u[mask_ci], v[mask_ci]
    ax.fill_betweenx(v, x1=u, facecolor='k', alpha=alphavalue)
    
    # no x- or yticks and labels
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_ticks_position('none')
    ax.set_ylim(ylim)
    ax.set_xlim(left=0)
    # add title
    ax.set_title('KDE', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # add horizontal line at mean T below KDE curve
    ax.hlines(mean, 0, x[np.abs(y-mean).argmin()], color='k', linestyle='dashed', linewidth=1)
    
    
def plot_boxplot(data, ax):
    flierprops = dict(marker='o', markerfacecolor='black', markersize=5, linestyle='none')
    ### PLOT ###
    ax = sns.boxplot(data['temp'], color='white', width=.4, flierprops=flierprops,
                        orient="v", linewidth=1.4)#, notch=True)

    plt.setp(ax.artists, edgecolor = 'k', facecolor='w') # b&w coloring
    plt.setp(ax.lines, color='k') # b&w coloring
    ax.set_xticks([]), ax.set_yticks([]) # remove ticks
    ax.set_ylabel(''), ax.set_yticklabels('') # remove ticklabels and axis label
    ax.set_title('boxplot', fontsize=16)


### Choose files ###
'''
Choose a file that contains temperature data (_temperature_estimates.txt)
and fitting data (_ifors_averaged_data.fit) from ifors. 
You can also specify several files using wildcards!
Below, enter the paths of the files to process.
'''
#temp_file_location = '/home/pgross/heiBOX/programming/ifors_analysis/testdata/PG192*_estimates.txt'
temp_file_location = '/media/pgross/gross-data/analytical_data/raman/graphitization/results/MO14/temperature_estimates.txt'
fit_file_location = '/home/pgross/heiBOX/programming/ifors_analysis/testdata/PG*_data.fit'

#write_results = input("Do you want the summary of results to be written to a file? Type y/n; default = n --> ")
write_results_to_file = False
#if write_results == 'y':
#    write_results_to_file = True

temp_aggregate = pd.DataFrame()
fit_aggregate = pd.DataFrame()


### analyze individual samples ###
for file in list(glob.glob(temp_file_location)):
    # load temperature data
    temp_data = load_temperature_file_as_dataframe(file)
    temp_aggregate = pd.concat([temp_aggregate, temp_data])
    
    # obtain sample temperature statistics
    samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci = sample_statistics(temp_data)
    
    # output results to txt-file
    if write_results_to_file == True:
        write_statistics_to_file(samplename, count, mean, stdev, minval, maxval, median, quart1, quart3, lower_ci, upper_ci)
    
    ### plot T-data of individual samples ###
    # Create a figure of given size
    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 1])
    
    ### Add a subplot
    ax0 = plt.subplot(gs[0])
    xlim, ylim = plot_temperatures(temp_data, samplename, mean, stdev, upper_ci, lower_ci, ax0)
    #plot_host_minerals(data_sorted, ax0)
    
    ### Add a subplot
    ax1 = plt.subplot(gs[1])
    plot_kde(temp_data, mean, stdev, upper_ci, lower_ci, ax1, xlim, ylim)
    
    ### Add a subplot
    #ax2 = plt.subplot(gs[2])
    #plot_boxplot(data, ax2)
    #plot_host_kde(data, ax2, xlim, ylim)
    
    plt.tight_layout()
    plt.show()
    
    #fig.savefig(samplename + '_result.png', format='png', bbox_inches='tight', dpi=300)
    #fig.savefig(samplename + '_result.pdf', format='pdf', bbox_inches='tight')


### analyze samples as bulk ###
for file in list(glob.glob(fit_file_location)):
    # load fitting data
    fit_data = load_fit_file_as_dataframe(file)
    fit_aggregate = pd.concat([fit_aggregate, fit_data])


# merge dataframes and calculate more parameters
aggregate = pd.merge_ordered(fit_aggregate, temp_aggregate, on=['sample', 'analysis'])
aggregate['T_STA'] = np.where(aggregate['G_shape_factor'] > 3, aggregate['G_STA'], aggregate['D_STA'])


### plot all aggregated sample data together ###
# as violinplot sorted by ascending temp
sns.violinplot(aggregate.sort_values('temp'), x='sample', y='temp')#, palette='magma')
#sns.violinplot(aggregate.sort_values('temp'), x='sample', y='T_STA')

# as boxplot sorted by ascending temp
#sns.boxplot(aggregate.sort_values('temp'), x='sample', y='temp')

# this is the calibration function for T from Lünsdorf et al. 2017:
#STA = 42.879 #variable
#T = -8.259*1E-5*STA**3 + 3.733*1E-2*STA**2 - 6.445*STA + 6.946*1e2

