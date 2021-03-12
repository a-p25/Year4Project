######################### Importing Packages #########################
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


# To run in hydra:
# First use command: export PATH=/usr/local/anaconda3/bin:$PATH
# To run with a logging file, use nohup python CNNScriptMk2.py & 

# Uses https://www.tensorflow.org/tutorials/keras/classification
######################### Organising the Data #########################

# Read Stellar Assembly hdf5 file and write data to a csv file
def hdf5ToCsv(filename = './stellar_assembly.hdf5', data=('StellarMassExSitu', 'StellarMassInSitu', 'StellarMassTotal'), csvFileName='inSituExSituTot.csv', show_log=False):
    f = h5py.File(filename, 'r')
    snapshots = f.keys() 
    snap_99 = f['Snapshot_99']
    data_lists = []
    for d in data:
         data_lists.append(np.array(snap_99[d]))
    # Values in list: ['StellarMassAfterInfall', 'StellarMassBeforeInfall', 'StellarMassExSitu', 'StellarMassFormedOutsideGalaxies', 'StellarMassFromCompletedMergers', 'StellarMassFromCompletedMergersMajor', 'StellarMassFromCompletedMergersMajorMinor', 'StellarMassFromFlybys', 'StellarMassFromFlybysMajor', 'StellarMassFromFlybysMajorMinor', 'StellarMassFromOngoingMergers', 'StellarMassFromOngoingMergersMajor', 'StellarMassFromOngoingMergersMajorMinor', 'StellarMassInSitu', 'StellarMassTotal']
    f.close()
    exSitu, inSitu, totMass = data_lists

    # Write Index, Ex Situ, In Situ, and Total Mass to csv file (index=subhalo id num)
    with open (csvFileName, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'ExSitu', 'InSitu', 'Tot'])
        for index, exSituVal in enumerate(exSitu):
            writer.writerow([index, exSituVal, inSitu[index], totMass[index]])

    if show_log == True:

        logging.info('Created CSV File with {} entries'.format(len(exSitu))) # 4371211 entries

    return csvFileName


def multiParamCSV(hdf5CSV, imgDirPath='Images', show_log=False):
    # Read mass data from CSV 
    stellarAssembly = pd.read_csv(hdf5CSV, index_col='Index')

    # Get list of subhalos in image directory
    filenames = os.listdir(imgDirPath)
    imgDirIDs = [int(''.join([c for c in filename if c.isdigit()])) for filename in filenames]

    # Retrieve only the mass data for subhalos in image directory
    stellarAssemblyFiltered = stellarAssembly.loc[imgDirIDs]

    # Create new column for ESMF
    stellarAssemblyFiltered['ESMF'] = stellarAssemblyFiltered['ExSitu']/stellarAssemblyFiltered['Tot']

    # Assign flags based on the ESMF
    # 0 :  ESMF <= 0.1
    # 1 : 0.1 < ESMF < 0.4
    # 2 : 0.4 <= ESMF <= 1

    esmfFlagDict = {0: 0.1,
                    1: 0.4,
                    2: 1 }

    stellarAssemblyFiltered.loc[stellarAssemblyFiltered['ESMF']<=esmfFlagDict[0], 'esmfFlag'] = 0
    stellarAssemblyFiltered.loc[(stellarAssemblyFiltered['ESMF']>esmfFlagDict[0]) & (stellarAssemblyFiltered['ESMF']<esmfFlagDict[1]), 'esmfFlag'] = 1
    stellarAssemblyFiltered.loc[stellarAssemblyFiltered['ESMF']>=esmfFlagDict[1], 'esmfFlag'] = 2

    # stellarAssemblyFiltered is a pandas dataframe with insitu, exsitu, totmass, and esmf flags

    # Get fits data for each subhalo
    for subhaloID, filename in zip(imgDirIDs, filenames):
        fitsData = fits.getdata(imgDirPath+'/'+filename)
        stellarAssemblyFiltered.loc[subhaloID, 'imgData']
    
    print(stellarAssemblyFiltered.head())


    return None


# Create a csv file of filenames and esmf only for files in image directory
def esmfFlags(outputFilename, dataCSV='inSituExSituTot.csv', imgDirPath='Images', show_log=False):
    # Load csv into a pandas dataframe
    stellarAssembly = pd.read_csv(dataCSV, index_col='Index')
    if show_log == True:
        logging.info('stellarAssembly Database columns: {}'.format(stellarAssembly.columns))
        logging.info('Type of stellarAssembly index values: {}'.format(type(stellarAssembly.index[0])))
        logging.info('First row of stellarAssembly data: \n{}'.format(stellarAssembly.loc[0]))

    # Retrieve subhalo IDs of images in image directory
    filenames = os.listdir(imgDirPath)
    imgDirIDs = [int(''.join([c for c in filename if c.isdigit()])) for filename in filenames]
    if show_log == True:
        logging.info('First 10 IDs from Image directory: {}'.format(imgDirIDs[:10]))
        logging.info('Type of first ID from Image directory: {}'.format(type(imgDirIDs[0])))
    
    # Filter dataframe to include only subhalos corresponding to ones in image directory
    stellarAssemblyFiltered = stellarAssembly.loc[imgDirIDs]
    if show_log == True:    
        logging.info('Filtered stellarAssembly dataframe created')

    # Create new column containing ESMF
    stellarAssemblyFiltered['ESMF'] = stellarAssemblyFiltered['ExSitu']/stellarAssemblyFiltered['Tot']
    if show_log == True:
        # logging.info('Filtered stellarAssembly with ESMF: \n{}'.format(stellarAssemblyFiltered))
        pass

    # Assign flags based on ESMF 
    # 0 : ESMF <= 1
    # 1 : 1 < ESMF < 0.9
    # 2 : ESMF >= 0.9
    esmfFlagDict = {0: 0.1,
                    1: 0.4,
                    2: 1 }
        
    stellarAssemblyFiltered.loc[stellarAssemblyFiltered['ESMF']<=esmfFlagDict[0], 'esmfFlag'] = 0

    stellarAssemblyFiltered.loc[(stellarAssemblyFiltered['ESMF']>esmfFlagDict[0]) & (stellarAssemblyFiltered['ESMF']<esmfFlagDict[1]), 'esmfFlag'] = 1

    stellarAssemblyFiltered.loc[stellarAssemblyFiltered['ESMF']>=esmfFlagDict[1], 'esmfFlag'] = 2

    if show_log == True:
        # logging.info('stellarAssembly wih flags: \n{}'.format(stellarAssemblyFiltered))
        logging.info('Flags assigned with following bounds:\nRegion 0: ESMF <= {}\nRegion 1: {} < ESMF < {}\nRegion 2: ESMF >= {}'.format(esmfFlagDict[0], esmfFlagDict[0], esmfFlagDict[1], esmfFlagDict[1]))
        count0flags = len((stellarAssemblyFiltered.loc[stellarAssemblyFiltered['esmfFlag']==0]))
        count1flags = len((stellarAssemblyFiltered.loc[stellarAssemblyFiltered['esmfFlag']==1]))
        count2flags = len((stellarAssemblyFiltered.loc[stellarAssemblyFiltered['esmfFlag']==2]))
        logging.info('Number of Subhalos in:\nRegion 0: {}\nRegion 1: {}\nRegion 2: {}'.format(count0flags, count1flags, count2flags))

    # Write subhalo Id, ESMF and ESMF Flag to csv file
    stellarAssemblyFiltered[['ESMF', 'esmfFlag']].to_csv(outputFilename)

    return stellarAssemblyFiltered

def createESMFDataset(esmfFlagsCSV, imgDirPath, show_log=False, training_split=0.8):
    esmfFlags = pd.read_csv(esmfFlagsCSV, index_col='Index') # Index = subhalo ID
    imgCount = len(esmfFlags.index)
    if show_log == True:
        logging.info('Subhalos in {}: {}'.format(esmfFlagsCSV, imgCount))

    #Find shape of first fits file in directory:
    exampleFits = fits.getdata(imgDirPath+'/'+'processed_broadband_{}.fits'.format(esmfFlags.index[0]))
    exampleShape = np.shape(exampleFits)

    # Make empty arrays with correct shape
    subhaloData = np.zeros(shape=(imgCount, exampleShape[0], exampleShape[1], exampleShape[2]))
    subhaloLabels = np.zeros(shape=(imgCount))
    subhaloIDs = np.zeros(shape=(imgCount))


    for i, subhaloID in enumerate(esmfFlags.index):
        # Retrieve data and label for subhalo
        filename = 'processed_broadband_{}.fits'.format(subhaloID)
        data = fits.getdata(imgDirPath+'/'+filename)
        label = esmfFlags['esmfFlag'][subhaloID]

        # Write data, label, and ID to arrays
        subhaloData[i] = data
        subhaloLabels[i] = label
        subhaloIDs[i] = subhaloID

    trainDataLabelIDs = [subhaloData, subhaloLabels, subhaloIDs]

    valDataLabelIDs = None
    
    # Split data into training and validation sets
    if training_split != 1:
        trainImgCount = round(training_split*imgCount)
        # Lists of training and validation data
        trainData, valData = np.split(trainDataLabelIDs[0], [trainImgCount])
        trainLabels, valLabels = np.split(trainDataLabelIDs[1], [trainImgCount])
        trainIDs, valIDs = np.split(trainDataLabelIDs[2], [trainImgCount])
        trainDataLabelIDs = [trainData, trainLabels, trainIDs]
        valDataLabelIDs = [valData, valLabels, valIDs]

    if show_log == True:
        logging.info('Image data shape: {}'.format(trainDataLabelIDs[0].shape))
        logging.info('Length of training data: {}'.format(len(trainDataLabelIDs[0])))
        logging.info('Length of validation data: {}'.format(len(valDataLabelIDs[0])))

    return trainDataLabelIDs, valDataLabelIDs

def createArrayTestingImgs(imgDirPath, show_log=False):

    # Get number of images in directory
    testingFilenames = os.listdir(imgDirPath) 
    imgCount = len(testingFilenames)

    # Get shape of images
    exampleFits = fits.getdata(imgDirPath+'/'+testingFilenames[0])
    exampleShape = np.shape(exampleFits)

    # Make empty arrays with correct shape
    testImgData = np.zeros(shape=(imgCount, exampleShape[0], exampleShape[1], exampleShape[2]))
    subhaloIDs = np.zeros(shape=(imgCount))

    # Crate array of image data
    for i, filename in enumerate(testingFilenames):
        subhaloID = int(''.join([c for c in filename if c.isdigit()]))

        # Retrieve data and label for subhalo
        data = fits.getdata(imgDirPath+'/'+filename)

        # Write data, label, and ID to arrays
        testImgData[i] = data
        subhaloIDs[i] = subhaloID

    return testImgData, subhaloIDs


def imageFigure(img, subhaloID=None, layer=0, augmented=False):
    # Layer is either 0, 1, 2 or 3 (the fits images have 4 layers)
    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    if augmented == True:
        plt.savefig('subhalo{}Layer{}Augmented.png'.format(str(subhaloID), str(layer)))
    else:
        plt.savefig('subhalo{}Layer{}.png'.format(str(subhaloID), str(layer)))

    return None

def dataAnalysis(trainData, show_log=False):
    # Plots 1 image and shows maximum value to check they are normalised to 1
    trainData, trainLabels, trainIDs = trainData
    fitsImage = trainData[0]
    label = trainLabels[0]
    subhaloID = trainIDs[0]
    if show_log == True:
        logging.info('Image of Subhalo {}'.format(str(subhaloID)))
    # for i, layer in enumerate(fitsImage):
        # if show_log == True:
        #     logging.info('Plotting Layer {}, with max value {}'.format(str(i), np.max(layer)))
        # plt.figure()
        # plt.imshow(fitsImage[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.savefig('ExampleFitsLayer{}_{}.png'.format(str(i), str(subhaloID)))

def randomAugmentation(img, show_log=False):
    # Rotate the image
    if show_log == True:
        imageFigure(img, augmented=False)
    
    layer0, layer1, layer2, layer3 = img
    randomTheta = np.random.uniform(10, 360)

    newlayer0 = rotate(layer0, randomTheta, reshape=False)
    newlayer1 = rotate(layer1, randomTheta, reshape=False)
    newlayer2 = rotate(layer2, randomTheta, reshape=False)
    newlayer3 = rotate(layer3, randomTheta, reshape=False)
    
    img = [newlayer0, newlayer1, newlayer2, newlayer3]

    # Flip the image
    img = np.fliplr(img)

    if show_log == True:
        imageFigure(newImg, augmented=True)

    return img

def classBalancing(trainingData, show_log=False):
    trainData, trainLabels, trainIDs = trainingData
    imgInputShape = np.shape(trainData[0])
    trainData = np.asarray(trainData)
    countTrainFlag0 = np.sum(trainLabels == 0)  
    countTrainFlag1 = np.sum(trainLabels == 1)  
    countTrainFlag2 = np.sum(trainLabels == 2)  

    if show_log == True:
        logging.info('Training set images with flag:\n0: {}\n1: {}\n2: {}'.format(countTrainFlag0, countTrainFlag1, countTrainFlag2))


    # Apply data augmentation so the classes are split more equally

    ### Flag 1 
    # Find difference in number of flag 0 images and flag 1 images
    countTrainDiff1 = countTrainFlag0 - countTrainFlag1
    # Find indices of images with the correct flag
    trainFlagIndices1 = np.where(trainLabels == 1)[0]

    # Augment images with the correct flag until the class has enough images
    for i in range(countTrainDiff1):
        # Use modulo to loop through the indexes for the flag
        index = trainFlagIndices1[(i % len(trainFlagIndices1))]

        # Retrieve original image and augment to newImg
        img = trainData[index]
        newImg = randomAugmentation(img)

        # Add new image and the corresponding label and id to the datasets
        trainData = np.append(trainData , [newImg], axis=0)
        trainLabels = np.append(trainLabels, trainLabels[index])
        trainIDs = np.append(trainIDs, trainIDs[index])

        if show_log == True:
            # logging.info('Index in trainData: {}'.format(index))
            # logging.info('Shape of trainData: {}'.format(np.shape(trainData)))
            # logging.info('Shape of trainLabels: {}'.format(np.shape(trainLabels)))
            # logging.info('Shape of trainIDs: {}'.format(np.shape(trainIDs)))
            pass

    ### Flag 2
    # Find difference in number of flag 0 images and flag 2 images
    countTrainDiff2 = countTrainFlag0 - countTrainFlag2
    # Find indices of images with the correct flag
    trainFlagIndices2 = np.where(trainLabels == 2)[0]

    # Augment images with the correct flag until the class has enough images
    for i in range(countTrainDiff2):
        # Use modulo to loop through the indexes for the flag
        index = trainFlagIndices2[(i % len(trainFlagIndices2))]

        # Retrieve original image and augment to newImg
        img = trainData[index]
        newImg = randomAugmentation(img)

        # Add new image and the corresponding label and id to the datasets
        trainData = np.append(trainData , [newImg], axis=0)
        trainLabels = np.append(trainLabels, trainLabels[index])
        trainIDs = np.append(trainIDs, trainIDs[index])

        if show_log == True:
            # logging.info('Index in trainData: {}'.format(index))
            # logging.info('Shape of trainData: {}'.format(np.shape(trainData)))
            # logging.info('Shape of trainLabels: {}'.format(np.shape(trainLabels)))
            # logging.info('Shape of trainIDs: {}'.format(np.shape(trainIDs)))
            pass


    if show_log == True:
        countTrainFlag0 = np.sum(trainLabels == 0)  
        countTrainFlag1 = np.sum(trainLabels == 1)  
        countTrainFlag2 = np.sum(trainLabels == 2)  

        logging.info('Training set images with flag:\n0: {}\n1: {}\n2: {}'.format(countTrainFlag0, countTrainFlag1, countTrainFlag2))

    return trainData, trainLabels, trainIDs

def createCNN(imgInputShape):





def trainNetwork(trainingData, validData, show_log=False, save=False):
    # Unpack data
    trainData, trainLabels, trainIDs = trainingData
    valData, valLabels, valIDs = validData

    imgInputShape = np.shape(trainData[0])
    if show_log==True:
        logging.info('Input image shape: {}'.format(imgInputShape))

    # Create the model - simple for debugging 
    # model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=imgInputShape),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(3, activation='softmax') ]) # Number of output classes = 3
    # model.compile(optimizer='adam',
            #   loss='sparse_categorical_crossentropy',
            #   metrics=['accuracy'])


    # Create model - complex model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(100, (3,3), activation='relu',
                    data_format='channels_first', input_shape=imgInputShape))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.MaxPool2D((2,2), data_format='channels_first'))
    model.add(tf.keras.layers.Conv2D(200, (2,2), data_format='channels_first', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.MaxPool2D((2,2), data_format='channels_first'))
    model.add(tf.keras.layers.Conv2D(200, (3,3), data_format='channels_first', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.MaxPool2D((2,2), data_format='channels_first'))
    model.add(tf.keras.layers.Conv2D(200, (2,2), data_format='channels_first', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if show_log == True:
        logging.info('Model created and compiled')

    
    model.summary()

    model.fit(trainData, trainLabels, epochs=10)
    if show_log == True:
        logging.info('Model fitted')
        logging.info('Checking model on validation set')

    valLoss, valAcc = model.evaluate(valData, valLabels, verbose=2)
    if show_log == True:
        logging.info('Validation Loss: {}    Accuracy: {}'.format(valLoss, valAcc))

    if save == True:
        model.save('TrainedModelComplex.h5')
    return model


if __name__ == '__main__':
    logging.info('################## Start of Script {} ##################'.format(__file__))
    logging.info('TensorFlow Version: {}'.format(tf.__version__))
    stellarAssemblyCSVFile = hdf5ToCsv(show_log=True) # inSituExSituTot.csv
    esmfFlags = esmfFlags(show_log=False)
    trainData, valData = createESMFDataset('esmfFlags.csv', 'Images')
    # dataAnalysis(trainData, show_log=True)
    trainImgs, trainLabels, trainIDs = classBalancing(trainData, show_log=True)
    trainData = [trainImgs, trainLabels, trainIDs]
    trainNetwork(trainData, valData, show_log=True)

    # New workflow for training on csv of multiple parameters - Leave until later
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns
    # stellarAssemblyCSVFile = hdf5ToCsv(show_log=True)
    # multiParamCSV(stellarAssemblyCSVFile, show_log=True)


    logging.info('################## Finished running {} ##################'.format(__file__))
    