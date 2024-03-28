# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:12:54 2023

@author: ggn
"""

from warnings import filterwarnings

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

import gpax
from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_patches, extract_subimages
import atomai as aoi
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import random

gpax.utils.enable_x64()


def remove_nan_elements(array1):
    array2 = []
    for element in array1:
        if np.isnan(element) == False:
            array2.append(element)
    return(array2)

def reverse_2D_y(img):
    img_yr = np.zeros(np.shape(img))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_yr[i, j] = img[(img.shape[0]-1)-i, j]
            
    return img_yr

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

def generate_dkl_train(data_f):
    x = data_f.loc[:,3]
    y = data_f.loc[:,2]
   
    x1 = np.asarray(x)
    y1 = np.asarray(y)

    xdata_f = remove_nan_elements(x1)
    ydata_f = remove_nan_elements(y1)
    
    x_int = []
    for elements in xdata_f:
        x_int.append(int(elements))
   
    return x_int, ydata_f

def oneD_to_2D(array_1D):
    length = int(len(array_1D)**0.5)
    array_op = np.reshape(array_1D, (length,length))
    array_op = np.asarray(array_op)
    return array_op

    
def get_coord_intersect_indices(coord_sub, coord_super):
    index_values = []
    for element in coord_sub:
        a = np.where(coord_super[:,0] == element[0])[0]
        b = np.where(coord_super[:,1] == element[1])[0]
    ind_element = list(set(a).intersection(set(b)))[0]
    index_values.append(ind_element)
    return index_values

def plot_result(indices, obj, file_name_f, iteration):
    os.chdir(r"C:\Users\Public\Ganesh\DKL_Experiments\aq_maps")
    plt.figure(figsize= (7,5))
    plt.scatter(indices[:, 1], indices[:, 0], s=32, c=obj, marker='s')
    next_point = indices[obj.argmax()]
    plt.scatter(next_point[1], next_point[0], marker='x', c='k')
    plt.title("Acquisition function values")
    #plt.show()
    savedata = np.transpose([indices[:, 1],  indices[:, 0], obj])
    np.savetxt(str(file_name_f) + '_aqfn_'+ str(iteration) +'.txt',savedata, delimiter = '\t', header = 'x \ty \t acq_fn')    
    plt.savefig(str(file_name_f) + '_aqmap_'+ str(iteration) +'.png', dpi = 300, bbox_inches = 'tight', pad_inches = 1.0)
    
def plotGP_predict(indices, obj_mean, obj_var, indices_m, y_measured, next_coord, file_name_f, iteration):
    os.chdir(r"C:\Users\Public\Ganesh\DKL_Experiments\aq_maps")
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14,6))
    
    
    a = ax[0].scatter(indices[:,1], indices[:,0], c=obj_mean, cmap='viridis', linewidth=0.2)
    ax[0].scatter(indices_m[:,1], indices_m[:,0], marker='s', c=y_measured, cmap='jet', linewidth=0.2)
    ax[0].scatter(next_coord[1], next_coord[0], marker='x', c='r')
    #ax[0].scatter(X_opt_GP[0, 0], X_opt_GP[0, 1], marker='o', c='r')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(a, cax=cax, orientation='vertical')
    ax[0].set_title('Objective mean map', fontsize=10)
    
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    #ax[0].axes.xaxis.set_visible(False)
    #ax[0].axes.yaxis.set_visible(False)
    #ax[0].colorbar(a)

    b = ax[1].scatter(indices[:,1], indices[:,0], c=obj_var, cmap='viridis', linewidth=0.2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b, cax=cax, orientation='vertical')
    ax[1].set_title('Objective variance map', fontsize=10)
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')    
    #ax[1].axes.xaxis.set_visible(False)
    #ax[1].axes.yaxis.set_visible(False)
    #ax[1].colorbar(b)
    
    plt.savefig(file_name_f + '_gpmap_'+ str(iteration) +'.png', dpi = 300, bbox_inches = 'tight', pad_inches = 1.0)
    
    savedata = np.transpose([indices[:, 1],  indices[:, 0], obj_mean, obj_var])
    np.savetxt(str(file_name_f) + '_gpmap_'+ str(iteration) +'.txt',savedata, delimiter = '\t', header = 'x \ty \t mean|t variance')    
    
def get_features(image, step, window_size):
    coordinates = get_coord_grid(image, step, return_dict= False)
    extracted_features = extract_subimages(image, coordinates, window_size)
    patches, coords, _ = extracted_features
    patches = patches.squeeze()
    
    n, d1, d2 = patches.shape
    features = patches.reshape(n, d1*d2)
    return features, coords




def dkl_prediction(index_values, y_measured, imgdata, step, window_size, scan_file_name):
    

    X, coords_all = get_features(imgdata, int(step), int(window_size))
    X_measured = X[index_values]
    X_unmeasured = np.delete(X, index_values, 0)
    
    y_measured_norm = (y_measured - np.min(y_measured)) / (np.max(y_measured) - np.min(y_measured))
    coords_unmeasured = np.delete(coords_all, index_values, 0)
    coords_measured = coords_all[index_values]
        
    data_dim = X_measured.shape[-1]
    
    exploration_steps = 1
    key1, key2 = gpax.utils.get_keys()
    
    for e in range(exploration_steps):
        print("{}/{}".format(e+1, exploration_steps))
        
        # update GP posterior
        dkl = gpax.viDKL(data_dim, 2, kernel = 'RBF')
        dkl.fit(  # you may decrease step size and increase number of steps (e.g. to 0.005 and 1000) for more stable performance
            key1, X_measured, y_measured_norm, num_steps=200, step_size=0.05)
        
        
        mean, var = dkl.predict(key2, X)
        # Compute UCB acquisition function
        
        power_f = (len(index_values) - 20)
        if power_f < 1:
            power_f = 0
            
        beta_iter = 10 * (0.9**(power_f))
        
        if beta_iter < 0.001:
            beta_iter = 0.001
    
            
        obj = gpax.acquisition.UCB(key2, dkl, X_unmeasured, beta = beta_iter, maximize=True)
        
        #obj = gpax.acquisition.UCB(key2, dkl, X_unmeasured, beta = 0.1, maximize=True)
        #obj = gpax.acquisition.EI(key2, dkl, X_unmeasured, maximize = True)
    
        # Select next point to "measure"  
        next_point_idx = obj.argmax()
        
        #index of the point in superset of indices
        next_index_super = get_coord_intersect_indices([coords_unmeasured[next_point_idx]], coords_all)[0]
        next_coord = coords_all[next_index_super]
    
         # Plot current result
        plot_result(coords_unmeasured, obj, scan_file_name, len(index_values)) 
        plotGP_predict(coords_all, mean, var, coords_measured, y_measured_norm, next_coord, scan_file_name, len(index_values))

    return int(next_index_super)
  
    
def dkl_gpax_LV(scan_file, compilation_file, step, window_size, iter_no):
    
    #Extract scan data
    os.chdir(r"C:\Users\Public\Ganesh\DKL_Experiments")
    data_scan = pd.read_csv(str(scan_file), delimiter = "\t", skiprows = 2, header = None)
    xdata, ydata, zdata = generate_xyz_arrays(data_scan)
    imgdata = oneD_to_2D(zdata)
    imgdata = reverse_2D_y(imgdata)
    
    
    #Get all the possible coordinate indices from the images
    #step = 10 
    #window_size = 30
    _, coords = get_features(imgdata, int(step), int(window_size))
    print(len(coords))
        
    #Extract training data
    m_indices = []
    y_measured = []
    
        


    
    # Random sampling in the intial iterations
    measure_index = random.randint(0, len(coords)-1)
    
    
    if iter_no > 20:
    # Call DKL to predict the next index to be measured      
          
        data_compilation = pd.read_csv(str(compilation_file), delimiter = "\t", skiprows = 1, header = None)
        #data_compilation = np.asarray(data_compilation)
        if len(data_compilation) > 0:
            m_indices, y_measured = generate_dkl_train(data_compilation)
            
            y_temp = []
            indices_temp =[]
            for element_index in range(len(y_measured)):
                if y_measured[element_index] > 0:
                    y_temp.append(y_measured[element_index])
                    indices_temp.append(m_indices[element_index])
            m_indices = indices_temp
            y_measured = y_temp
        
        
        dkl_index = dkl_prediction(m_indices, y_measured, imgdata, step, window_size, scan_file)
        measure_index = int(dkl_index)
    
 
    r_array = []
     
    element = coords[measure_index]
         
    #the notation of x and y are opposite for the DKL implementation
    r_array.append(int(element[1]))
    r_array.append(int(element[0]))
     
    r_array.append(int(measure_index))
    r_array.append(int(len(coords)))
     
    return r_array   
      
      
      