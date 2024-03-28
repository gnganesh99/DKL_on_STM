# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:06:55 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def positive_channel_area(v, i):
    
    v_s = []
    curr_s = []
    scalar = 0
    
    for k in range(len(v)):
        if v[k] > 0:
            v_s.append(v[k])
            curr_s.append(i[k])
    
    scalar += np.trapz(curr_s, x = v_s)
    
    return scalar
    
def spectrum_area_range(v, i, v_initial, v_final):
    v_s = []
    curr_s = []
    scalar = 0
    
    for k in range(len(v)):
        if v[k] >= v_initial and v[k] <= v_final:
            
            v_s.append(v[k])
            curr_s.append(i[k])
    
    scalar += np.trapz(curr_s, x = v_s)
    
    
    return scalar
    
def array_within_threshold(x, y, th):
    y = np.asarray(y)
    y_norm = (y - np.min(y))/(np.max(y)-np.min(y))
    x_th = []
    y_th = []
    for k in range(len(x)):
        if y_norm[k] <= th:
            x_th.append(x[k])
            y_th.append(y_norm[k])
            
        else:
            break
        
    return x_th, y_th


def partial_linear_fit(v_f, c_f, v_range):    
    dv = v_f[1]-v_f[0]
    points =  int(v_range/dv)
    #print(points)
    score = 0
    i_offset = 0
    for i in range(len(v_f)-points):
        v_fit = v_f[i:i+points]
        c_fit = c_f[i:i+points]
        X = np.asarray(v_fit).reshape((-1, 1))
        reg = LinearRegression(fit_intercept = True).fit(X, c_fit)
        if reg.score(X, c_fit) >= score and reg.coef_[0] > 0.1:
            score = reg.score(X, c_fit)
            i_offset = i

    v_fit = v_f[i_offset:i_offset+points]
    c_fit = c_f[i_offset:i_offset+points]
    X = np.asarray(v_fit).reshape((-1, 1))
    reg = LinearRegression(fit_intercept = True).fit(X, c_fit)
    
    if reg.coef_ > 0:
        x_inter = (- reg.intercept_/reg.coef_)
    else:
        x_inter = [-4]
    
    v_line = np.linspace(x_inter, v_f[-1], 50)
    X = np.asarray(v_line).reshape((-1, 1))
    y_pred = reg.predict(X)
    return y_pred, v_line, reg.coef_, reg.intercept_, x_inter, score


def bandgap_slopefit(v, c, V_range):
    v = np.asarray(v)
    c = pd.Series(c).rolling(window = 3,  min_periods=1, closed = 'both').mean()
    c = np.asarray(c)
        
    v_p = []
    c_p = []
    v_n = []
    c_n = []

    for k in range(len(v)):
        if v[k]< 0:
            v_n.append(abs(v[k]))
            if c[k] < 0:
                c_n.append(0)
            else:
                c_n.append(c[k])

        if v[k]>= 0:
            v_p.append(v[k])
            if c[k] < 0:
                c_p.append(0)
            else:
                c_p.append(c[k])

    v_n = v_n[::-1]
    c_n =c_n[::-1]

    #print(v_n, c_n)
    x_inter_n = [0]
    x_inter_p = [0]
    v_norm_th = 0.2
    
    if len(v_n) > 2:
        v_neg, c_neg = array_within_threshold(v_n, c_n, v_norm_th)   
        #print(len(v_n), len(c_n), v_neg, c_neg)
        
        if len(v_neg) > 2:
            y_pred_p, x_pred_n, slope, y_inter, x_inter_n, score = partial_linear_fit(v_neg, c_neg, V_range)
    
    if len(v_p) >2:
        v_pos, c_pos = array_within_threshold(v_p, c_p, v_norm_th)
        #print(v_pos, c_pos)
        
        if len(v_pos) > 2:
            y_pred_p, x_pred_p, slope, y_inter, x_inter_p, score = partial_linear_fit(v_pos, c_pos, V_range)
    
    bandgap = (x_inter_n[0]+x_inter_p[0])
    
    return bandgap, x_inter_n[0], x_inter_p[0]

def positive_slopefit(v, c, V_range):
    v = np.asarray(v)
    c = pd.Series(c).rolling(window = 3,  min_periods=1, closed = 'both').mean()
    c = np.asarray(c)
        
    v_p = []
    c_p = []

    for k in range(len(v)):

        if v[k]>= 0:
            v_p.append(v[k])
            if c[k] < 0:
                c_p.append(0)
            else:
                c_p.append(c[k])

    x_inter_p = [0]
    offset = [0]
    v_norm_th = 0.1   
    if len(v_p) >2:
        v_pos, c_pos = array_within_threshold(v_p, c_p, v_norm_th)
        #print(v_pos, c_pos)
        y_pred_p, x_pred_p, slope, y_inter, x_inter_p, score = partial_linear_fit(v_pos, c_pos, V_range)
    
    bandgap = (offset[0]+x_inter_p[0])
    
    return bandgap, x_inter_p[0]

def negative_slopefit(v, c, V_range):
    v = np.asarray(v)
    c = pd.Series(c).rolling(window = 3,  min_periods=1, closed = 'both').mean()
    c = np.asarray(c)
    
    v_n = []
    c_n = []

    for k in range(len(v)):
        if v[k]<= 0:
            v_n.append(abs(v[k]))
            if c[k] < 0:
                c_n.append(0)
            else:
                c_n.append(c[k])



    v_n = v_n[::-1]
    c_n =c_n[::-1]

    #print(v_n, c_n)
    x_inter_n = [0]
    offset = [0]
    v_norm_th = 0.2
    
    if len(v_n) > 2:
        v_neg, c_neg = array_within_threshold(v_n, c_n, v_norm_th)   
        #print(len(v_n), len(c_n), v_neg, c_neg)
        y_pred_p, x_pred_n, slope, y_inter, x_inter_n, score = partial_linear_fit(v_neg, c_neg, V_range)
    
    
    bandgap = (x_inter_n[0]+offset[0])
    
    return bandgap, x_inter_n[0]



def scalarizer(spectroscopy_filename, spec_array, x_pixel, y_pixel):
        
    
    os.chdir(r"C:\Users\Public\Ganesh\DKL_Experiments")
    file_name = str(spectroscopy_filename)
    
    
    
    v = spec_array[0]
    channel = 2  # 1 = current, 2 = LIX, 3 = LIY. 4-6 is for the backward direction

        
    #scalar = positive_channel_area(v, spec_array[channel])
    
    scalar1 = spectrum_area_range(v, spec_array[channel], 0, 0.5)
    scalar2 = spectrum_area_range(v, spec_array[channel], 0, 1.5)
    scalar = 5 + (scalar1/scalar2)*1000
    
    #scalar = 15000 - scalar
    
    
    #scalar = np.exp(1 - spec_array[channel][0])
    
    #_, scalar = negative_slopefit(v, spec_array[channel], 0.2)
    #scalar, _, _ = bandgap_slopefit(v, spec_array[channel], 0.2)
        
    
    
    
    #if scalar > 0:
    #scalar = 15000 - scalar
    
    
        
        
        
    '''
    data_param = pd.read_csv(file_name, delimiter = "\t", skiprows = 0, nrows = 1, header = None)
    coords = np.asarray(data_param)[0]
    
    data = pd.read_csv(file_name, delimiter = '\t', skiprows = 1, header = None,)
    v = np.asarray(data.loc[:,0])
    i1 = np.asarray(data.loc[:,1])
    i2 = np.asarray(data.loc[:,2])
    
    
    v_s = []
    i1_s = []
    i2_s = []

    for k in range(len(v)):
        if v[k] > 0:
            v_s.append(v[k])
            i1_s.append(i1[k])
            i2_s.append(i2[k])
   
    scalar = (np.trapz(i1_s, x = v_s) + np.trapz(i2_s, x = v_s))/2
    '''
    file1 = open(file_name + "_alldata.txt", "a")  # append mode
    savearray = np.transpose(spec_array)
    #savearray = spec_array
    #file1.write('\n'+str(savearray))
    file1.write('\n\n'+str(x_pixel)+'\t'+str(y_pixel)+'\t'+str(scalar)+'\nvoltage\ti_forward\ti_backward\n'+str(savearray))
    file1.close()
    
    
    return scalar