# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:06:10 2020

@author: Jack

A consolidation of all the key functions defined so far.
Will need to edit this from time to time as some are path specific.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import random
import pickle
plt.style.use('classic')


def SKIRT_matcher(assembly_data_fnames, SKIRT_fnames_directory):
    '''Extracts stellar assembly info and matches it with subhalo IDs.
    Returns 2d array with isd, esd, total and id as rows.'''
    #read data
    assembly_data = np.loadtxt(assembly_data_fnames)
    #file strucutre is [isd, esd, total]
    SKIRT_fnames = os.listdir(SKIRT_fnames_directory)
    assembly_data = np.transpose(assembly_data)
    
    #get image IDs which correspond to positions in stellar assembly arrays
    index_array = []
    for i in SKIRT_fnames:
        counter = 0
        while i[counter] != '.':
            counter += 1
        index_array.append(int(i[20:counter]))
    
    #get stellar assembly info for SKIRT images using IDs found above
    SKIRT_assembly_data = [assembly_data[i] for i in index_array]
    SKIRT_assembly_data = np.transpose(SKIRT_assembly_data)
    SKIRT_assembly_data = np.transpose(np.array([SKIRT_assembly_data[0], SKIRT_assembly_data[1], SKIRT_assembly_data[2], index_array]))    
    return SKIRT_assembly_data
    
    #plot selection of images based on limits selected by user
    #this code also displays the objects' ESMFs and total masses

def image_plotter(ESMF_Range, Total_Range, SKIRT_assembly_data, path):
    
    """
    Plots N images where N is the user input in a nice format.
    User may define total mass range and ESMF range for which to plot images.
    This is useful for checking data makes sense.
    """
    
    ESMF = [i[1]/i[2] for i in SKIRT_assembly_data]
    total = [i[2] for i in SKIRT_assembly_data]
    ID = [i[3] for i in SKIRT_assembly_data]
    
    cut_ESMF = [ESMF[i] for i in range(len(ESMF)) if Total_Range[0] < total[i] < Total_Range[1] and ESMF_Range[0] < ESMF[i] < ESMF_Range[1]]
    cut_total = [total[i] for i in range(len(ESMF)) if Total_Range[0] < total[i] < Total_Range[1] and ESMF_Range[0] < ESMF[i] < ESMF_Range[1]]
    cut_ID = [ID[i] for i in range(len(ESMF)) if Total_Range[0] < total[i] < Total_Range[1] and ESMF_Range[0] < ESMF[i] < ESMF_Range[1]]
    fnames = [path + r'\processed_broadband_' + str(int(i)) + '.fits' for i in cut_ID]
    
    print(f'There are {len(fnames)} objects in this sample. Type how many you would like to display.')
    lim = int(input())
    
    cut_ESMF, cut_total, cut_ID = cut_ESMF[0:lim], cut_total[0:lim], cut_ID[0:lim]
    
    if np.sqrt(len(cut_ID))%1 != 0:
        fig_side_length = int(np.sqrt(len(cut_ID)))+1
    if np.sqrt(len(cut_ID))%1 == 0.0:
        fig_side_length = int(np.sqrt(len(cut_ID)))
    
    fig = plt.figure(figsize=[5*fig_side_length, 5*fig_side_length])
    ax = fig.subplots(fig_side_length, fig_side_length)
    
    for i in range(fig_side_length):
        counter = 0
        while counter < fig_side_length:
            if i*fig_side_length+counter < len(cut_ID):
                hdul = fits.open(fnames[i*fig_side_length+counter])
                print(i,i*fig_side_length+counter)
                ax[i][counter].imshow(np.arcsinh(hdul[0].data[0]), cmap = 'gray')
                ax[i][counter].set_title('ESMF = ' + str(np.round(cut_ESMF[i*fig_side_length+counter], 4))+' Total Mass = ' + str(np.round(cut_total[i*fig_side_length+counter],4)))
                counter +=1
            else:
                counter += 1
                
def data_loader(data_directory, assembly_data_path, class_boundary, save_as = None):
   
    """
    Generates testing and training data, testing and training labels and 
    corresponding auxilliary data for a given class boundary. save_as option
    allows user to save code to path given as the save_as variable.
    """
    matched_data = SKIRT_matcher(assembly_data_path, data_directory)
    
    print('matched data')
    
    training_data, training_labels, testing_data, testing_labels = [], [], [], []
    aux_training_data, aux_testing_data = [], []
    
    for i in range(len(matched_data)):
        #print(data_directory + r'\processed_broadband_' + str(int(i[3])) + '.fits')
        path = data_directory + r'\processed_broadband_' + str(int(matched_data[i][3])) + '.fits'
        data = np.arcsinh(fits.open(path)[0].data[0])
        data = np.reshape(data, [len(data), len(data), 1])
        
        if i%10 == 0:
            testing_data.append(data)   
            aux_testing_data.append(matched_data[i])
            if matched_data[i][1]/matched_data[i][2] < class_boundary:
                testing_labels.append(0)
            else:
                testing_labels.append(1)
                
        else:
            training_data.append(data)
            aux_training_data.append(matched_data[i])
            if matched_data[i][1]/matched_data[i][2] < class_boundary:
                training_labels.append(0)
            else:
                training_labels.append(1)
        
        if i%500 == 0:
            print(i, ' out of ', len(matched_data))
        
    if save_as:
        np.save(save_as, np.array([np.array(training_data), np.array(training_labels), np.array(testing_data), np.array(testing_labels), aux_testing_data, aux_training_data]), allow_pickle=True)
    
    print('generated testing and training')
        
    return np.array(training_data), np.array(training_labels), np.array(testing_data), np.array(testing_labels), aux_testing_data, aux_training_data

def convolve_PSF(image):
    """
    Convolves a synthetic image with a Gaussian point spread function (PSF) to
    reproduce the effects of telescope optics and atmospheric noise on the 
    image.
    
    Parameters:
    image : array of shape (channels, N, N), assuming that the first value is
        the number of filters (channels)
        
    Returns:
    convolved_image : array of shape (channels, N, N) with the result of the
        convolution
    """
    channels = np.shape(image)[0] # assume channels are listed first
    # create an array to store results of convolution
    convolved_image = np.zeros(shape=np.shape(image))
    for channel in range(channels):
        # default size of PSF kernel is 8 times std in each direction
        # originally defined std in both x, y axes; then changed to single
            # value (same for both axes) to work with astropy 2.0.9
        psf = Gaussian2DKernel(std_pix[channel])
        convolved_image[channel] = convolve(image[channel], psf)
    return convolved_image


def noise_modelling(image, final_size, noise_background, noise_std):
    """
    Incorporates sky background noise into a synthetic image. The noise is 
    sampled from a Gaussian distribution for which the mean and the
    standard deviation can be specified.
    
    Parameters:
    image : array of shape (channels, N, N), assuming that the first value is
        the number of filters (channels)
    final_size : integer M >= N: processed images will have shape 
        (channels, M, M)
    noise_background : array with mean values of the noise distribution for 
        each filter
    noise_std : array with standard deviations of the noise distribution for
        each filter
    
    Returns:
    noisy_image : array of shape (channels, N, N) with the noise contribution
    """
    channels = np.shape(image)[0] # assume channels are listed first
    # create array to store results of noise modelling and padding
    noisy_image = np.zeros(shape=(channels, final_size, final_size))
    for channel in range(channels):
        initial_size = np.shape(image[channel])[-1]
        if final_size == initial_size:
            noisy_image[channel] += image[channel]
        # padding is done slightly differently depending on whether the
            # difference between the initial/final image size is even or odd
        elif (final_size - initial_size) % 2 == 0: # even case
            noisy_image[channel] = np.pad(image[channel], int((
                final_size-initial_size)/2), mode='constant',constant_values=0)
        else: # odd case
            noisy_image[channel] = np.pad(image[channel], (int((
                final_size-initial_size)/2 - 0.5), int((
                final_size-initial_size)/2 + 0.5)), mode='constant', 
                constant_values=0)
        # add noise sampled from a normal distribution to the padded image
        noisy_image[channel] += np.random.normal(noise_background[channel], 
                                 noise_std[channel], (final_size, final_size))
    noisy_image[noisy_image < 0] = 0 # set all values below 0 to 0
    return noisy_image


def image_postprocessing(image, final_size, noise_background, noise_std):
    """
    Carries out the post processing steps required to prepare a synthetic image
    for use in the Convolutional Neural Network model. Calls the convolve_PSF
    and noise_modelling functions as well as ensuring each 2D array of data
    has values between 0 and 1.
    
    Parameters:
    image : array of shape (channels, N, N), assuming that the first value is
        the number of filters (channels)
    final_size : integer M >= N: processed images will have shape 
        (channels, M, M)
    noise_background : array with mean values of the noise distribution for 
        each filter
    noise_std : array with standard deviations of the noise distribution for
        each filter
    
    Returns:
    processed_image : array of shape (channels, N, N) corresponding to the
        processed image
    """
    convolved_image = convolve_PSF(image)
    noisy_convolved_image = noise_modelling(convolved_image, final_size, 
                                            noise_background, noise_std)
    channels = np.shape(noisy_convolved_image)[0]
    # create array to store results of rescaling of data
    processed_image = np.zeros(shape=(channels, final_size, final_size))
    for channel in range(channels):
        processed_image[channel] = noisy_convolved_image[channel]/np.max(
            noisy_convolved_image[channel])
    return processed_image