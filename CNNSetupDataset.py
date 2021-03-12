import logging
logging.basicConfig(level=logging.INFO) # For debugging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Too many future warnings being annoying

import h5py
import numpy as np
import csv
import os
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('agg') # Needed for running on hydra otherwise plt.figure() causes errors
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import rotate
import math
from CNNScriptMk2 import hdf5ToCsv, esmfFlags

######################### Configuration #########################
logging.info('################## Start of Script: {} ##################'.format(__file__))
# Path to training/validation folder
trainImgPath = './ImagesFullSet/trainValDir'
testImgPath = './ImagesFullSet/testDir'
# Path to stellar assembly file, including file name
stellarAssemblyPath = './stellar_assembly.hdf5'
file = h5py.File(stellarAssemblyPath, 'r')

######################### Data #########################
# Save a csv file containing Index, Ex Situ, In Situ, and Total Mass (dest = 'inSituExSituTot.csv')
stellarAssemblyCSV = hdf5ToCsv(filename = stellarAssemblyPath, show_log=True)

######## Training Data
# Create a csv containing image filenames, ESMF, and ESMF flags  (dest = 'esmfFlags.csv')
esmfFlags('esmfFlags.csv', dataCSV='inSituExSituTot.csv', imgDirPath=trainImgPath, show_log=True)

# Data augmentation (saving to separate csv) (dest = 'augmentedEsmfFlags.csv') 
# Add in later

######## Testing Data
# Save a csv file for the testing data only
esmfFlags('testingEsmfFlags.csv', dataCSV='inSituExSituTot.csv', imgDirPath=testImgPath, show_log=True)

logging.info('################## End of Script: {} ##################'.format(__file__))