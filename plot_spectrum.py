#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:56:28 2021

@author: philip
"""

import matplotlib.pyplot as plt
import numpy as np
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

start, end = 70, 600
x1, y1 = load_spectrum('./data/qz_rruff.txt', start, end)
x2, y2 = load_spectrum('./data/alm_rruff.txt', start, end)
x3, y3 = load_spectrum('./data/prp_rruff.txt', start, end)
x4, y4 = load_spectrum('./data/grs_rruff.txt', start, end)

plt.plot(x1, y1)
#plt.plot(x2+x3+x4, y2)
#plt.plot(x2, y2)
#plt.plot(x3, y3)
#plt.plot(x4, y4)