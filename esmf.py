import csv
import os
import numpy as np
import pandas as pd


def subhaloIDs(dirNames, imageDir = 'Images'):
    '''
    Generates a list of subhalo id numbers for each image directory
    '''
    namesPlusIDs = []
    for direc in dirNames:
        filenames = os.listdir(imageDir+'/'+direc)
        for filename in filenames:
            if 'broadband' in filename:
                pass
            else:
                filenames.remove(filename)

        idNums = [''.join([c for c in filename if c.isdigit()]) for filename in filenames]
        
        namesPlusIDs.append([filenames, idNums])

    return namesPlusIDs


def assignLabels(csvfile, subhalos, esmf_bounds, thresholdTotMass=0):
    '''
    For the image filenames and id numbers, get the ex situ mass fraction and categorise based on this. Only uses subhalos below a threshold total mass (in 10^10 solar mass units).
    '''

    # Assign the subhalo file names and ID numbers to lists
    test, train, val = subhalos
    testNames, testIDs = (test[0], [np.int64(idNum) for idNum in test[1]])
    trainNames, trainIDs = (train[0], [np.int64(idNum) for idNum in train[1]])
    valNames, valIDs = (val[0], [np.int64(idNum) for idNum in val[1]])

    # Printing  stuff for debugging
    print('Number of subhalo images in directories: {}'.format((len(testNames)+len(trainNames)+len(valNames))))

    csvData = pd.read_csv(csvfile)
    
    # Filter out subhalos that are higher than the total mass threshold, if there is one
    if thresholdTotMass != 0:
        # massFilter = (csvData['Tot']<=thresholdTotMass) & (csvData['Tot']!=0)
        massFilter = (csvData['Tot']<=thresholdTotMass)
        csvData = csvData[massFilter]

    # Convert the panda dataframes to lists for easier comparison with the image ids
    subhaloIndex = (csvData['Index'].tolist())
    exSitu = np.array(csvData['Ex-Situ'].tolist())
    # inSitu = np.array(csvData['In-Situ'].tolist())
    tot = np.array(csvData['Tot'].tolist())

    # Collect the id numbers and esmf for each subhalo in the testing directory, provided the total mass is below the threshold
    test_image_esmf = []
    for num in testIDs:
        try: 
            index = subhaloIndex.index(num)
            halo_data = [num, exSitu[index]/tot[index]]
            test_image_esmf.append(halo_data)
        except ValueError: # If subhalo has been cut out of data due to mass threshold
            pass

    # Collect the id numbers and esmf for each subhalo in the training directory, provided the total mass is below the threshold
    train_image_esmf = []
    for num in trainIDs:
        try: 
            index = subhaloIndex.index(num)
            halo_data = [num, exSitu[index]/tot[index]]
            train_image_esmf.append(halo_data)
        except ValueError: # If subhalo has been cut out of data due to mass threshold
            pass

    # Collect the id numbers and esmf for each subhalo in the validation directory, provided the total mass is below the threshold
    val_image_esmf = []
    for num in valIDs:
        try: 
            index = subhaloIndex.index(num)
            halo_data = [num, exSitu[index]/tot[index]]
            val_image_esmf.append(halo_data)
        except ValueError: # If subhalo has been cut out of data due to mass threshold
            pass

    # Printing stuff for debugging
    total_images = len(train_image_esmf)+len(test_image_esmf)+len(val_image_esmf)
    if thresholdTotMass!=0:
        print('Number of subhalos in directories with total mass below threshold {}: {}'.format(thresholdTotMass, total_images))
    else:
        print('Number of subhalos in directories: {}'.format(total_images))

    # Assign flags to the subhalos depending on the bounds of each category
    
    flags = range(len(esmf_bounds))
    
    # Testing set
    test_flags = []
    for halo in test_image_esmf:
        halo_esmf = halo[1]
        halo_id = halo[0]
        for i in flags:
            if halo_esmf > esmf_bounds[i][0] and halo_esmf <= esmf_bounds[i][1]:
                test_flags.append([halo_id, i])
            
    # Training set
    train_flags = []
    for halo in train_image_esmf:
        halo_esmf = halo[1]
        halo_id = halo[0]
        for i in flags:
            if halo_esmf > esmf_bounds[i][0] and halo_esmf <= esmf_bounds[i][1]:
                train_flags.append([halo_id, i])

    # Test set
    val_flags = []
    for halo in val_image_esmf:
        halo_esmf = halo[1]
        halo_id = halo[0]
        for i in flags:
            if halo_esmf > esmf_bounds[i][0] and halo_esmf <= esmf_bounds[i][1]:
                val_flags.append([halo_id, i])

    with open('test_esmf_flags.csv', 'w', newline='') as testfile:
        writer = csv.writer(testfile)
        writer.writerows(test_flags)

    with open('train_esmf_flags.csv', 'w', newline='') as trainfile:
        writer = csv.writer(trainfile)
        writer.writerows(train_flags)

    with open('val_esmf_flags.csv', 'w', newline='') as valfile:
        writer = csv.writer(valfile)
        writer.writerows(val_flags)

    return None


# testIDs = subhaloIDs(['test_dir'])
namesPlusIDs = subhaloIDs(('test_dir','train_dir','val_dir'))
(test, train, val) = namesPlusIDs


esmf_bounds = [[0, 0.02], [0.02, 0.98], [0.98, 1]]
assignLabels('insitu_exsitu_tot.csv', (test, train, val), esmf_bounds)