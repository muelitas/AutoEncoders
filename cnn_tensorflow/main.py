'''/**************************************************************************
    File: main.py
    Author: Mario Esparza
    Date: 05/19/2021
    
    I am following these two tutorials:
    https://blog.keras.io/building-autoencoders-in-keras.html
    https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
    
    Note: I am padding all audios to the same size (using the longest audio as
    the size to pad to).
***************************************************************************'''
import gc
import random
import sys

import numpy as np
import tensorflow as tf

from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import RMSprop
from utils import SC_DATASET, get_spectrograms, AutoEncoder
from utils import plot_spectrogram, DataGenerator

SC_path = '/media/mario/GRAYUSB'
folders_of_interest = ['backward', 'bird']
num_of_files = 50 #number of files to grab from each folder
split_dev = 0.2 #80% train, 20% validation
split_test = 0.5 #from 20%, split 10% and 10% for validation and testing
RANDOM_SEED = 42

#Hyper Parameters
HP = {  
      #'cnn1_filters': [8],
      #'cnn1_kernel': [3],
      #'cnn1_stride': [cnstnt.CNN_STRIDE],
      #'gru_dim': [64], #for now, same as n_mels
      'gru_hid_dim': 64,
      #'gru_layers': [2],
      #'gru_dropout': [0.6],
      #'n_class': [-1], #dynamically initialized later
      'mels': 128,
      #'dropout': [0.4], #classifier's dropout
      'e_0': 1e-4, #initial learning rate
      #'T': [35], #Set to -1 if you want a steady LR throughout training
      'bs': 1, #batch size
      'epochs': 2,
      'sr': 16000,
      'inChannel' : 1, #required for first CNN layer
      #These two values are used to mimic torchaudio.transforms.MelSpectrogram
      'N': 448, #length of the FFT window, originally, it was 400
}
HP['HL'] = HP['N'] // 2 #hop length

#Set same random seed for all modules
random.seed(RANDOM_SEED) #For random
# tf.random.set_seed(RANDOM_SEED) #For tensorflow
seed(RANDOM_SEED) #For numpy

#Get Dataset (paths to audios) and size of the audio with most samples 
dataset = SC_DATASET(SC_path, folders_of_interest, num_of_files)
longest = dataset.get_longest_size(HP['sr'])
#Make sure that all audios' spectrograms return the same shape. Also, get
#dimensions that will be used below in {Input}.
HP['dim1'], HP['dim2'] = dataset.check_audios(longest, HP)

#Split into training, validation and testing
train_paths, dev_paths = train_test_split(dataset, test_size=split_dev)
dev_paths, test_paths = train_test_split(dev_paths, test_size=split_test)

# Initialize generators
#TODO are these generators ignoring the last batch if odd-batch?
train_gen = DataGenerator(train_paths, HP, longest, False)
dev_gen = DataGenerator(dev_paths, HP, longest, False)
test_gen = DataGenerator(test_paths, HP, longest, False)

input_img = Input(shape = (dim1, dim2, HP['inChannel']))
autoencoder = Model(input_img, AutoEncoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

# autoencoder.summary()

#New way of doing it (with DataGenerators)
#.fit() doesn't support validation data being a generator
autoencoder.fit(train_gen,
                shuffle=True, #To shuffle before each epoch
                epochs = HP['epochs'],
                batch_size = HP['bs'],
                validation_data = dev_gen,
                workers=2)



decoded_specs = autoencoder.predict(test_gen, batch_size = HP['bs'])
N = 2
for i in range(0, N):
    plot_spectrogram(np.squeeze(test_X[i]), HP['sr'])
    plot_spectrogram(np.squeeze(decoded_specs[i]), HP['sr'])
'''    

#TODO, make sure you can reproduce results (check that random seed is working)

#Collect garbage (clear space in memory)
gc.collect()