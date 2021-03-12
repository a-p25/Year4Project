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
from CNNScriptMk2 import createESMFDataset, trainNetwork, createArrayTestingImgs
from datetime import datetime

logging.info('################## Start of Script: {} ##################'.format(__file__))
now = datetime.now()
logging.info('Script start time: {}'.format(now))

testImgPath = './ImagesFullSet/testDir'

reconstructed_model = tf.keras.models.load_model("TrainedModelComplex.h5")

testingImgData, testingSubhaloIDs = createArrayTestingImgs(testImgPath, show_log=False)

predictions = reconstructed_model.predict(testingImgData)
now = datetime.now()
logging.info('Finish predictions: {}'.format(now))

predictionsDataframe = pd.DataFrame(predictions, columns = ['Column_0','Column_1','Column_2'], index = testingSubhaloIDs)
predictionsDataframe.to_csv("predictionsComplexModel.csv", index=True)

now = datetime.now()
logging.info('End time: {}'.format(now))
logging.info('################## End of Script: {} ##################'.format(__file__))
