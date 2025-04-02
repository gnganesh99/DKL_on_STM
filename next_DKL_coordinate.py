# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:42:02 2023

@author: ggn
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_patches, extract_subimages
import atomai as aoi



def get_features(image, step, window_size):
    coordinates = get_coord_grid(image, step, return_dict= False)
    extracted_features = extract_subimages(image, coordinates, window_size)
    patches, coords, _ = extracted_features
    patches = patches.squeeze()
    
    n, d1, d2 = patches.shape
    features = patches.reshape(n, d1*d2)
    return features, coords

def remove_nan_elements(array1):
    array2 = []
    for element in array1:
        if np.isnan(element) == False:
            array2.append(element)
    return(array2)


def generate_xyz_arrays(data_f):
    x = data_f.loc[:,0]
    y = data_f.loc[:,1]
    z = data_f.loc[:,2]
   

    x1 = np.asarray(x)
    y1 = np.asarray(y)
    z1 = np.asarray(z)
   

    xdata_f = remove_nan_elements(x1)
    ydata_f = remove_nan_elements(y1)
    zdata_f = remove_nan_elements(z1)
    
    return xdata_f, ydata_f, zdata_f

def patch_points(x_array, y_array, xc, yc, frame):
    x_offset = xc-frame/2
    y_offset = yc - frame/2
    
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    
    x_array = x_array - x_offset*np.ones(len(x_array))
    y_array = y_array - y_offset*np.ones(len(y_array))
    count = 0
    
    for i in range(len(x_array)):
        if x_array[i] > 0 and x_array[i] <= frame:
            if y_array[i] > 0 and y_array[i] <= frame:
                count += 1
    
    return count

def patch_weight(x_array, y_array, weight, xc, yc, frame):
    x_offset = xc-frame/2
    y_offset = yc - frame/2
    
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    
    x_array = x_array - x_offset*np.ones(len(x_array))
    y_array = y_array - y_offset*np.ones(len(y_array))
    net_weight = 0
    
    for i in range(len(x_array)):
        if x_array[i] > 0 and x_array[i] <= frame:
            if y_array[i] > 0 and y_array[i] <= frame:
                net_weight += weight[i]
          
    return net_weight


def next_scan_coords(file_name, pixels, frame_old, frame_new):
    
    
    dummy_image = np.ones(shape = (pixels, pixels))
    frame = int(pixels * frame_new/ frame_old)
    #print(frame)
    _, coords = get_features(dummy_image, 10, frame)
    
    points_in_frame = 0
    next_ind = 0

    data= pd.read_csv(str(file_name), delimiter = "\t", skiprows = 1, header = None)
    x_pixels, y_pixels, scalar = generate_xyz_arrays(data)

    for i in range(len(coords)):
        x0 = coords[i][1]
        y0 = coords[i][0]
        points = patch_points(x_pixels, y_pixels, x0, y0, frame)
        if points > points_in_frame:
            points_in_frame = points
            next_ind = i
            
    x_next = (coords[next_ind][1]/pixels)*frame_old
    y_next = (coords[next_ind][0]/pixels)*frame_old
    
    r_array = []
    
    r_array.append(x_next)
    r_array.append(y_next)
    

    return r_array

def next_scan_coordWeight(file_name, pixels, frame_old, frame_new):
    
    
    dummy_image = np.ones(shape = (pixels, pixels))
    
    frame =  int(pixels* frame_new/ frame_old)
    _, coords = get_features(dummy_image, 10, frame)
    
    net_score = 0

    data= pd.read_csv(str(file_name), delimiter = "\t", skiprows = 1, header = None)
    x_pixels, y_pixels, scalar = generate_xyz_arrays(data)

    for i in range(len(coords)):
        x0 = coords[i][1]
        y0 = coords[i][0]
        score = patch_weight(x_pixels, y_pixels, scalar, x0, y0, frame)
        if score > net_score:
            net_score = score
            next_ind = i
            
    x_next = (coords[next_ind][1]/pixels)*frame_old
    y_next = (coords[next_ind][0]/pixels)*frame_old

    r_array = []
     
    r_array.append(x_next)
    r_array.append(y_next)
     

    return r_array

