# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:41:15 2020

@author: Jack

A script using general_model.py, the generated training data and a list of 
250 sets of hyperparamters to generate networks for all 1250 possible
combinations.
"""
import general_model

hyperparamter_list = np.load(r'/net/lnx0/scratch/jxa737/data/hp_list.npy',allow_pickle = True)
test_data_fnames = ['0.02.npy', '0.03.npy', '0.06.npy', '0.1.npy', '0.16.npy']

for i in test_data_fnames:
    index = 0
    for j in hyperparamter_list:
        
        path = r'/net/lnx0/scratch/jxa737/data/testing and training data/' + i
        data = np.load(path, allow_pickle=True)
        training_data = data[0]
        training_labels = data[1]
        testing_data = data[2]
        testing_labels = data[3]
        print('test and train data loaded')
        
        model = general_model.General.Model(*j)
        print('model loaded')
        
        model.train(training_data, training_labels, testing_data, testing_labels, 20, 
                r'/net/lnx0/scratch/jxa737/CNNs/best network test ' + str(index) + ' ' + i[:-4] + '.ckpt', 
                r'/net/lnx0/scratch/jxa737/CNNs/best network info '+ str(index) + ' ' + i[:-4] +, class_balancer=True)
        print('model trained')
        
        index += 1