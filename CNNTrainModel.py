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
from CNNScriptMk2 import createESMFDataset, trainNetwork
from datetime import datetime

logging.info('################## Start of Script: {} ##################'.format(__file__))
now = datetime.now()
logging.info('Start time: {}'.format(now))
######################### Configuration #########################

# Name of csv file containing image filenames, ESMF, and ESMF flags
flagsCSV = 'esmfFlags.csv'

# Path to training and validation directory
imgPath = './ImagesFullSet/trainValDir'

######################### Prepare Dataset #########################
trainData, valData = createESMFDataset(flagsCSV, imgPath)
logging.info('ESMF dataset imported from {} and {}'.format(flagsCSV, imgPath))
######################### Train the Model #########################
logging.info('Beginning training network')
now = datetime.now()
logging.info('Training start time: {}'.format(now))
trainNetwork(trainData, valData, show_log=True, save=True)
logging.info('Model trained and saved')

now = datetime.now()
logging.info('End time: {}'.format(now))
logging.info('################## End of Script: {} ##################'.format(__file__))