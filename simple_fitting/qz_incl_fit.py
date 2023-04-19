#!/usr/bin/env python
# to clear variable space
#%reset -f

'''
This script fits functions to measured Raman spectra of quartz inclusions.
It critically REQUIRES the lmfit package. Documentation and download at:
    https://lmfit.github.io/lmfit-py/index.html
Cite lmfit as:
    Newville, Matthew, Stensitzki, Till, Allen, Daniel B., & 
    Ingargiola, Antonino. (2014, September 21). LMFIT: 
    Non-Linear Least-Square Minimization and Curve-Fitting for 
    Python (Version 0.8.0). Zenodo. http://doi.org/10.5281/zenodo.11813

INPUTS:
    - measured spectrum file (.txt) as delivered by LabSpec software.
        The data must not contain spurious peaks and steep background.
        You can also use wildcards for multi-file processing.
    - change the required user input below for your needs
    
OUTPUTS:
    - inputfilename_fit.txt: List of fitted peak parameters.
    - inputfilename_fit.png: Image of fitted spectrum with components.

Tested with Python 3.6 and lmfit 0.9
written by Philip Groß, April 2020
'''

##  --------------------------------------------------------------------- ##
###   PREAMBLE   ###
# load packages:
#import sys
import os
import glob
import numpy as np
#from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit.models import LinearModel, PseudoVoigtModel
from lmfit import Parameters
plt.close('all')

def load_spectrum(filename, lower_bound, upper_bound):
    """load spectrum file, crop data for given wavenumber limits,
    rescale intensities to 0-100"""
    data = np.loadtxt(filename)
    
    # if needed, reorder data to begin with low wavenumbers
    if data[0,0] > data[-1,0]:
        x = np.flipud(np.array(data[:,0]))
        y = np.flipud(np.array(data[:,1]))
    else:
        x = np.array(data[:,0])
        y = np.array(data[:,1])
        
    # consistency checks
    if lower_bound >= upper_bound:
        print('ERROR: Lower bound is larger than upper bound!')
    if lower_bound < min(x):
        lower_bound = min(x)
    if upper_bound > max(x):
        upper_bound = max(x)

    # find indexes (idx) of wnum where upper and lower bound are exceeded
    idx = np.argwhere((x < upper_bound) & (x > lower_bound))

    # convert idx to scalars:
    upr = np.squeeze(idx[0])
    lwr = np.squeeze(idx[-1])

    # crop input data to specified extent
    x = x[upr:lwr]
    y = y[upr:lwr]

    # rescale ydata
    y = y/max(y)*100    
    
    return x, y
##  --------------------------------------------------------------------- ##


##------------------------------------------------------------------------##
###   USER INPUT   ###

# Fitting region:
fitting_region = '464' # peak region to fit; '464', '206' or '128'

# specify file location; wildcards allowed; use \\ in Windows (see below)
file_location = '../data/PG320_grt3_qz--001.txt'
#file_location = '/home/pgross/Documents/projekte/kim/inclusions/PG349T/grt5_neu/*.txt'

# Plot results and print to file?
plot_results = 'yes'    # 'yes' or 'no'
print_results = 'yes'   # 'yes' or 'no'
# to get confidence intervals for all peak parameters:
detailed_report = 'no'  # 'yes' or 'no'; SLOW!

# if below this value, correlation data is ignored during output
min_correl = 0.5

# uncertainty for the residuals
sigma = 2
##------------------------------------------------------------------------##

# evaluation interval, in wavenumbers
if fitting_region == '464':
    start, end = 420, 530
elif fitting_region == '206':
    start, end = 150, 280
elif fitting_region == '128':
    start, end = 90, 160
else:
    print('ERROR: Wrong fitting region!')
    

# open files
for file in list(glob.glob(file_location)):
    print(file)
    filename = os.path.basename(file)
    samplename = os.path.splitext(filename)[0]
    print(samplename)

    # load data
    x, y = load_spectrum(file, start, end)
    
    ### initialize model
    params = Parameters()
    
    # define model
    if fitting_region == '464':
        # initialize model parameters
        params = Parameters()
        
        # linear background
        background = LinearModel(prefix='background_')
        params.add('background_intercept', value=min(y))
        params.add('background_slope', value=0)
        
        # qz464 peak
        qz464 = PseudoVoigtModel(prefix='qz_')
        params.add('qz_center', value=464, min=455, max=480)
        params.add('qz_amplitude', value=max(y))
        params.add('qz_sigma', value=5, min=1, max=50)
        params.add('qz_fraction', value=0.5, vary=True, min=0.01, max=1.0)
        
        # grt450 peak
        #grt450 = PseudoVoigtModel(prefix='grt_')
        #params.add('grt_center', value=450, min=446, max=454)
        #params.add('grt_amplitude', value=max(y))
        #params.add('grt_sigma', value=5, min=1, max=50)
        #params.add('grt_fraction', value=0.5, vary=True, min=0.01, max=1.0)
        
        model = background + qz464 #+ grt450
        
    elif fitting_region == '206':
        # linear background
        background = LinearModel(prefix='background_')
        params.add('background_intercept', value=min(y))
        params.add('background_slope', value=0)
        
        # qz206 peak
        qz206 = PseudoVoigtModel(prefix='qz_')
        params.add('qz_center', value=206, min=195, max=230)
        params.add('qz_amplitude', value=max(y))
        params.add('qz_sigma', value=5, min=1, max=50)
        params.add('qz_fraction', value=0.5, vary=True, min=0.01, max=1.0)
        
        model = background + qz206
        
    elif fitting_region == '128':
        # linear background
        background = LinearModel(prefix='background_')
        params.add('background_intercept', value=min(y))
        params.add('background_slope', value=0)
        
        # qz128 peak
        qz128 = PseudoVoigtModel(prefix='qz_')
        params.add('qz_center', value=128, min=120, max=140)
        params.add('qz_amplitude', value=max(y))
        params.add('qz_sigma', value=5, min=1, max=50)
        params.add('qz_fraction', value=0.5, vary=True, min=0.01, max=1.0)
        
        model = background + qz128
    
    else:
        print('ERROR: Wrong fitting region!')
    
    # show initial values
    #params.pretty_print()
    
    # initialize model
    init = model.eval(params, x=x)
    
    # fit model
    out = model.fit(y, params, x=x)
    
    # print fitting report
    #print(out.fit_report(min_correl=min_correl))
    
    # print detailed ci_report
    if detailed_report == 'yes':
        print(out.ci_report())
    
    # read-out fitted model components
    components = out.eval_components(x=x)
    
    # write results to ordered dict. to access values, do e.g.:
    # out.params.valuesdict()['qz_center']
    #outdict = out.params.valuesdict()
    print('peak center position: ', round(out.params.valuesdict()['qz_center'], 2))
    # print fitted peak center position:
    
    
    # write statistics to file 
    if print_results == 'yes':
        if detailed_report == 'yes':
            with open((samplename + '_fit.txt'), 'w') as f_out:
                f_out.write(samplename + '\n' 
                            + out.fit_report(min_correl=min_correl)  
                            + '\n' + '\n' + 'CI_report:'
                            + out.ci_report())
                
        else:
            with open((samplename + '_fit.txt'), 'w') as f_out:
                f_out.write(samplename + '\n' + out.fit_report(min_correl=min_correl))
    

    ###   PLOTTING   ###
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    
    # data points
    ax0.plot(x, y, marker='.', color='k', markersize=5, linestyle='none', label='data')
    
    # best fit
    ax0.plot(x, out.best_fit, color='tab:red', linestyle='dashed', label='best fit', alpha=0.8)
    
    # initial fit
    #ax0.plot(x, out.init_fit, color='k', label='initial fit', alpha=0.7)
    
    # uncertainty band
    d = out.eval_uncertainty(sigma=sigma)
    ax0.fill_between(x, out.best_fit-d, out.best_fit+d, color='r', alpha=0.2,
                     label=(str(sigma) + '$\sigma$ uncertainty band'))
    
    # components                 
    #ax0.plot(x, components['background_'], 'darkcyan', alpha=0.5, label='background')
    #ax0.plot(x, components['qz_'], 'royalblue', alpha=0.5, label='qz')

#    ax0.plot(x, components['background_'], 'darkcyan', alpha=0.5, label='background')
#    ax0.plot(x, components['qz_'], 'royalblue', alpha=0.5, label='qz')
#    ax0.plot(x, components['grt495_'], 'slateblue', alpha=0.5, label='grt495')
    
    # axis styling
    ax0.set_ylabel('intensity (a.u.)')
    ax0.set_title(samplename)
    ax0.set_xticklabels([])
    ax0.legend()
    
    # residuals
    ax1 = plt.subplot(gs[1])
    out.plot_residuals(yerr=out.eval_uncertainty(sigma=sigma), ax=ax1)
    ax1.set_title('')
    #ax1.set_title('Residuals ($\pm 2 \sigma$)')
    #ax1.get_title().remove()
    #ax1.get_legend().remove()
    ax1.set_ylabel(str(sigma) + '$\sigma$ residuals (a.u.)')
    ax1.set_xlabel('wavenumber ($cm^{-1}$)')
    ax1.lines[1].set_markersize(3)
    ax1.lines[0].set_color('k')
    plt.tight_layout()
    
    # save figure
    if plot_results == 'yes':
        fig.savefig(samplename +'_fit.pdf')#, resolution=150)

    #plt.close('all')
