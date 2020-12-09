# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:14:11 2020

@author: Jack
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import tensorflow as tf
from tensorflow.keras import layers, models
import toolkit
import pickle
from zipfile import ZipFile
plt.style.use('default')
#plt.style.use('seaborn-colorblind')

class General:
    
    def __init__(self, checkpoint_path, metadata_path):
                
        """
        Allows user to create intance of model from saved parameters.
        Inputs are:
            checkpoint_path: The path to the checkpoint file saved when training. String.
            metadata: Path to additional data saved when training the model. String.
        """
        useful_metadata = np.load(metadata_path, allow_pickle = True)[3:]
        self.model = General.Model(*useful_metadata)
        self.model.model.load_weights(checkpoint_path)
        
    def performance_summary(self, testing_and_training_path, class_boundary = None, confusion_matrix = None):
        
        """
        Takes testing and training .npy file and performs network performance
        evaluation on data plotting various useful figures. This method will
        be edited frequently depending on specific investigations.
        """
        
        data = np.load(testing_and_training_path, allow_pickle = True)
        testing_data = data[2]
        testing_labels = data[3]
        aux_testing_data = data[4]
        
        model_results = self.model.model.predict(testing_data)
        
        correct_aux_data = [aux_testing_data[i] for i in range (len(model_results)) if np.argmax(model_results[i]) == testing_labels[i]]
        incorrect_aux_data = [aux_testing_data[i] for i in range (len(model_results)) if np.argmax(model_results[i]) != testing_labels[i]]
        
        cor_total = [i[2] for i in correct_aux_data]
        cor_ESFM = [i[1]/i[2] for i in correct_aux_data]
        
        incor_total = [i[2] for i in incorrect_aux_data]
        incor_ESFM = [i[1]/i[2] for i in incorrect_aux_data]
        
        fig = plt.figure(figsize = [10,10])
        ax = fig.gca()
        ax.plot(incor_total, incor_ESFM, '.', label = 'Incorrectly classified objects')
        ax.plot(cor_total, cor_ESFM, '.', label = 'Correctly classified objects')
        ax.set_xlabel('Total Mass')
        ax.set_ylabel('ESMF')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if class_boundary:
            ax.hlines(class_boundary ,10, 0.1, label = 'Class Boundary')
            ax.set_title('Class Boundary = ' + str(class_boundary) +', '+ 'Accuracy = ' + str(len(cor_total)/len(model_results)))
            
        ax.legend()
        
        if confusion_matrix:
            cor_above = len([i for i in range(len(model_results)) if np.argmax(model_results[i]) == 1 and testing_labels[i] == 1])
            cor_below = len([i for i in range(len(model_results)) if np.argmax(model_results[i]) == 0 and testing_labels[i] == 0])
            incor_above = len([i for i in range(len(model_results)) if np.argmax(model_results[i]) == 1 and testing_labels[i] == 0])
            incor_below = len([i for i in range(len(model_results)) if np.argmax(model_results[i]) == 0 and testing_labels[i] == 1])
            
            string = 'Class Boundary = ' + str(class_boundary) + '\n' + '========================================\n' + '                    ' + 'Model Above Boundary   Model Below Boundary\n' + 'Testing Set Above Boundary   ' + str(cor_above) + '                    ' +  str(incor_above) + '\n' 'Testing Set Below Boundary   ' + str(incor_below) + '                    ' +  str(cor_below) + '\n' + '========================================'
            
            print(string)

    class Model:
        
        def __init__(self, num_cov_layers, layer_depths, kernel_sizes, pooling_sizes, num_ful_layers, ful_layer_sizes, dropout, learning_rate, batch_size):
            """
            Initialize model based on input parameters.
            This allows many networks to be generated allowing hyperparameter
            optimization by multiple runs over a grid of hyperparameters.
            
            Kwargs are:
                num_cov_layers: Number of convolutional layers in model. Integer
                layer_depths: depth of output space. List of integers.
                kernel_sizes: Size of convolution kernals. List of tuples each with two values.
                pooling_sizes: Window size over which to perfom pooling. List of tuples each with two values.
                num_ful_layers: Number of fully connected layers. Integer
                ful_layer_sizes: Number of nodes for each fully connected layer. List of integers.
                dropout: Dropout to be applied between each layer. Float between 0 and 1.
                learning_rate: Initial learning rate for Adam optimizer. Float
                batch_size: Batch size for fitting. Integer
                class_boundary: Boudary between ESMF classes. Float between 0 and 1. Choose this value considering class imbalence.
            """
            self.num_cov_layers, self.layer_depths, self.kernel_sizes, self.pooling_sizes, self.num_ful_layers, self.ful_layer_sizes, self.dropout, self.learning_rate, self.batch_size = num_cov_layers, layer_depths, kernel_sizes, pooling_sizes, num_ful_layers, ful_layer_sizes, dropout, learning_rate, batch_size
             
            model = models.Sequential()
            model.add(layers.Conv2D(layer_depths[0], kernel_sizes[0], activation='relu', input_shape=(200, 200, 1)))
            
            if num_cov_layers > 1:
                for i in range(num_cov_layers-1):
                    if dropout != None:
                        model.add(layers.Dropout(dropout))
                    model.add(layers.MaxPooling2D(pooling_sizes[i]))
                    model.add(layers.Conv2D(layer_depths[i+1], kernel_sizes[i+1], activation='relu'))
            
            model.add(layers.Flatten())
                     
            for i in range(num_ful_layers):
                if dropout != None:
                     model.add(layers.Dropout(dropout))
                model.add(layers.Dense(ful_layer_sizes[i], activation = 'relu'))
            
            if dropout != None:
                model.add(layers.Dropout(dropout))
                
            model.add(layers.Dense(2))
            
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            
            model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
            
            self.model = model
            
        def summary(self):
            return self.model.summary()
            
        def train(self, training_data, training_labels, testing_data, testing_labels, epochs, save_best_network = False, save_training_info = False, class_balancer = False):
            
            """
            Trains network and saves best network and run info as requested by 
            user.
            """
            
            if class_balancer == True:
                class_0 = [training_data[i] for i in range(len(training_data)) if training_labels[i] == 0]
                class_1 = [training_data[i] for i in range(len(training_data)) if training_labels[i] == 1]
                
                class_0_temp = class_0
                class_1_temp = class_1
                
                #do 0 index larger than 1 index
                
                if len(class_0) > len(class_1):
                    frac = float(len(class_0))/float(len(class_1))
                    times_to_duplicate = int(frac)-1
                    tail_index = int(len(class_1)*(frac%1))
                    if frac > 2:
                        for i in range(times_to_duplicate):
                            class_1 = np.concatenate((class_1, class_1_temp))
                        class_1 = np.concatenate((class_1, class_1_temp[:tail_index]))
                    else:
                        class_1 = np.concatenate((class_1, class_1_temp[:tail_index]))
                    
                    print(len(class_0), len(class_1))
                    
                    training_data = np.concatenate((class_0, class_1))
                    training_labels = np.concatenate((np.zeros(len(class_0)), np.ones(len(class_1))))
                    
                    print(len(training_data), len(training_labels))
                
                #do 1 index larger than 0 index
                
                elif len(class_1) > len(class_0):
                    frac = float(len(class_1))/float(len(class_0))
                    times_to_duplicate = int(frac)-1
                    tail_index = int(len(class_0)*(frac%1))
                    print(tail_index)
                    if frac > 2:
                        for i in range(times_to_duplicate):
                            class_0 = np.concatenate((class_0, class_0_temp))
                        print(np.shape(class_0), np.shape(class_0_temp[:tail_index]))
                        class_0 = np.concatenate((class_0, class_0_temp[:tail_index]))
                    else:
                        class_0 = np.concatenate((class_0, class_0_temp[:tail_index]))
                    
                    print(len(class_0), len(class_1))
                    
                    training_data = np.concatenate((class_0, class_1))
                    training_labels = np.concatenate((np.zeros(len(class_0)), np.ones(len(class_1))))
                    
                    print(len(training_data), len(training_labels))
                
                #now do same for validation data
                
                class_0 = [testing_data[i] for i in range(len(testing_data)) if testing_labels[i] == 0]
                class_1 = [testing_data[i] for i in range(len(testing_data)) if testing_labels[i] == 1]
                
                class_0_temp = class_0
                class_1_temp = class_1
                
                #do 0 index larger than 1 index
                
                if len(class_0) > len(class_1):
                    frac = float(len(class_0))/float(len(class_1))
                    times_to_duplicate = int(frac)-1
                    tail_index = int(len(class_1)*(frac%1))
                    if frac > 2:
                        for i in range(times_to_duplicate):
                            class_1 = np.concatenate((class_1, class_1_temp))
                        class_1 = np.concatenate((class_1, class_1_temp[:tail_index]))
                    else:
                        class_1 = np.concatenate((class_1, class_1_temp[:tail_index]))
                    
                    print(len(class_0), len(class_1))
                    
                    testing_data = np.concatenate((class_0, class_1))
                    testing_labels = np.concatenate((np.zeros(len(class_0)), np.ones(len(class_1))))
                    
                    print(len(testing_data), len(testing_labels))
                
                #do 1 index larger than 0 index
                
                elif len(class_1) > len(class_0):
                    frac = float(len(class_1))/float(len(class_0))
                    times_to_duplicate = int(frac)-1
                    tail_index = int(len(class_0)*(frac%1))
                    print(tail_index)
                    if frac > 2:
                        for i in range(times_to_duplicate):
                            class_0 = np.concatenate((class_0, class_0_temp))
                        print(np.shape(class_0), np.shape(class_0_temp[:tail_index]))
                        class_0 = np.concatenate((class_0, class_0_temp[:tail_index]))
                    else:
                        class_0 = np.concatenate((class_0, class_0_temp[:tail_index]))
                    
                    print(len(class_0), len(class_1))
                    
                    testing_data = np.concatenate((class_0, class_1))
                    testing_labels = np.concatenate((np.zeros(len(class_0)), np.ones(len(class_1))))
                    
                    print(len(testing_data), len(testing_labels))
                        
                        
            
            if save_best_network and save_training_info:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_best_network, monitor = 'val_accuracy', save_best_only = True, save_weights_only=True, verbose=1)
                history = self.model.fit(training_data, training_labels, batch_size = self.batch_size, epochs=epochs, validation_data=(testing_data, testing_labels), callbacks = [cp_callback])
                
                #training info to save
                accuracy = history.history['accuracy']
                val_accuracy = history.history['val_accuracy']
                array_to_save = np.array([accuracy, val_accuracy, np.max(val_accuracy), self.num_cov_layers, self.layer_depths, self.kernel_sizes, self.pooling_sizes, self.num_ful_layers, self.ful_layer_sizes, self.dropout, self.learning_rate, self.batch_size])
                np.save(save_training_info, array_to_save, allow_pickle=True)
            
            elif save_best_network:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_best_network, monitor = 'val_accuracy', save_best_only = True, save_weights_only=True, verbose=1)
                history = self.model.fit(training_data, training_labels, batch_size = self.batch_size, epochs=epochs, validation_data=(testing_data, testing_labels), callbacks = [cp_callback])
                
            elif save_training_info:
                accuracy = history.history['accuracy']
                val_accuracy = history.history['val_accuracy']
                array_to_save = np.array([accuracy, val_accuracy, np.max(val_accuracy), self.num_cov_layers, self.layer_depths, self.kernel_sizes, self.pooling_sizes, self.num_ful_layers, self.ful_layer_sizes, self.dropout, self.learning_rate, self.batch_size])
                np.save(save_training_info, array_to_save, allow_pickle=True)
                
            else:
                history = self.model.fit(training_data, training_labels, batch_size = self.batch_size, epochs=epochs, validation_data=(testing_data, testing_labels))
        
#test model below
        
#define parameter ranges for grid search


# dropouts = [0, 0.1, 0.2, 0.3, 0.4]
# num_conv_layers = [1, 2, 3, 4]
# conv_layer_depths = [1, 10, 64]
# num_ful_layers = [1,2,3]
# learning_rate = [0.0001, 0.001, 0.01, 0.1]
# batch_size = [20, 32, 50, 100]
test_data_fnames = ['0.02.npy', '0.03.npy', '0.06.npy', '0.1.npy', '0.16.npy']
#test_data_fnames = ['0.1.npy', '0.16.npy']
# kernel_sizes = [(2,2), (3,3), (4,4), (5,5)]
# pooling_sizes = [(2,2), (3,3), (4,4), (5,5)]
# ful_con_sizes = [10, 100, 1000]
 

# =============================================================================
# for i in test_data_fnames:
#     
#     path = r'C:\\Users\\Jack\\Documents\\Uni\\Year_4\\Project\\data\\testing and training data\\' + i
#     data = np.load(path, allow_pickle=True)
#     training_data = data[0]
#     training_labels = data[1]
#     testing_data = data[2]
#     testing_labels = data[3]
#     print('test and train data loaded')
#     
#     model = General.Model(1, [32], [(3,3)], [], 1, [32], 0.1, 0.001, 64)
#     print('model loaded')
#     
#     model.train(training_data, training_labels, testing_data, testing_labels, 20, 
#             r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\first tests\best network test ' + i[:-4] + '.ckpt', 
#             r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\first tests\best network info3 '+ i[:-4], class_balancer=True)
#     print('model trained')
# =============================================================================



test1 = General(r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network test 0.1.ckpt',r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network info3 0.1.npy')
test1.performance_summary(r'C:\Users\Jack\Documents\Uni\Year_4\Project\data\testing and training data\0.1.npy', class_boundary=0.095, confusion_matrix=True)
test2 = General(r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network test 0.02.ckpt',r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network info3 0.02.npy')
test2.performance_summary(r'C:\Users\Jack\Documents\Uni\Year_4\Project\data\testing and training data\0.02.npy', class_boundary=0.02, confusion_matrix=True)
test3 = General(r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network test 0.03.ckpt',r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network info3 0.03.npy')
test3.performance_summary(r'C:\Users\Jack\Documents\Uni\Year_4\Project\data\testing and training data\0.03.npy', class_boundary=0.0336, confusion_matrix=True)
test4 = General(r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network test 0.06.ckpt',r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network info3 0.06.npy')
test4.performance_summary(r'C:\Users\Jack\Documents\Uni\Year_4\Project\data\testing and training data\0.06.npy', class_boundary=0.0566, confusion_matrix=True)
test5 = General(r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network test 0.16.ckpt',r'C:\Users\Jack\Documents\Uni\Year_4\Project\CNNs\scaled class boundary tests 4\best network info3 0.16.npy')
test5.performance_summary(r'C:\Users\Jack\Documents\Uni\Year_4\Project\data\testing and training data\0.16.npy', class_boundary=0.16, confusion_matrix=True)

 
        
        