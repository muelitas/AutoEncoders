'''/**************************************************************************
    File: main.py
    Author: Mario Esparza
    Created on: 05/22/2021
    Last edit on: 05/22/2021
    
    I used the help of these two tutorials:
    https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    https://blog.keras.io/building-autoencoders-in-keras.html
    
    Notes:
    - Only using 'backward' and 'bird' folders from SpeechCommands.
    - I am padding all audios to the same size (using the longest audio as
    the size to pad to).
    - To use tensorboard, run this {tensorboard --logdir=/tmp/autoencoder}
    in another terminal and uncomment 'callbacks' on .fit(). To monitor,
    access this link {http://0.0.0.0:6006}.
    - Results can be reproduced (i.e. random seed is working)
    
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
from utils import SC_DATASET, AutoEncoder
from utils import plot_spectrogram, DataGenerator

SC_path = '/media/mario/GRAYUSB'
folders_of_interest = ['backward', 'bird']
num_of_files = 20 #number of files to grab from each folder
split_dev = 0.2 #80% train, 20% validation
split_test = 0.5 #from 20%, split 10% and 10% for validation and testing
RANDOM_SEED = 42

#Hyper Parameters
HP = {
      'mels': 128,
      'bs': 2, #batch size
      'e_0': 3e-4, #notation from "Deep Learning Book"
      'epochs': 2,
      'sr': 16000,
      #N and HL values are used to mimic torchaudio.transforms.MelSpectrogram
      'N': 448, #length of the FFT window, originally, it was 400
}
HP['HL'] = HP['N'] // 2 #hop length

#Set same random seed for all modules
random.seed(RANDOM_SEED) #For random
tf.random.set_seed(RANDOM_SEED) #For tensorflow
seed(RANDOM_SEED) #For numpy

#Get Dataset (paths to audios) and size of the audio with most samples 
dataset = SC_DATASET(SC_path, folders_of_interest, num_of_files)
longest = dataset.get_longest_size(HP['sr'])
#Make sure that all audios' spectrograms return the same shape. Also, get
#dimensions that will be used below in {Input}.
HP['dim1'], HP['dim2'] = dataset.check_audios(longest, HP)

print(f"dim1 = {HP['dim1']}, dim2=HP['dim2']")

#Split into training, validation and testing
train_paths, dev_paths = train_test_split(dataset, test_size=split_dev)
dev_paths, test_paths = train_test_split(dev_paths, test_size=split_test)

# Initialize generators
train_gen = DataGenerator(train_paths, HP, longest, case='train', shuffle=True)
dev_gen = DataGenerator(dev_paths, HP, longest, case='dev', shuffle=True)
test_gen = DataGenerator(test_paths, HP, longest, case='test', shuffle=False)

#Initialize model
inpuT= Input(shape = (HP['dim1'], HP['dim2']), batch_size=HP['bs'])
autoencoder = Model(inpuT, AutoEncoder(inpuT, HP))
optimizer = tf.keras.optimizers.Adam(learning_rate=HP['e_0'])
autoencoder.compile(optimizer=optimizer, loss='mse')

autoencoder.summary()

#Train and validate
autoencoder.fit(train_gen,
                verbose=1,
                shuffle=False,
                epochs = HP['epochs'],
                batch_size = HP['bs'],
                validation_data = dev_gen,
                workers=1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                )

#Predict and plot "Predictions vs Originals"
decoded_specs = autoencoder.predict(test_gen, batch_size = HP['bs'])
N = min(5, len(decoded_specs))
for i in range(0, N):
    idx = i % 2
    batch_idx = i // 2
    #Plot original spectrogram
    plot_spectrogram(
        test_gen[batch_idx][0][idx].squeeze(), 
        HP['sr'],
        f"Original Spectrogram, Batch:{batch_idx}, idx:{idx}"
    )
    #Plot predicted spectrogram
    plot_spectrogram(
        np.squeeze(decoded_specs[i]), 
        HP['sr'],
        f"Predicted Spectrogram, i:{i}"
    )

#Release global state and collect garbage (clear gpu memory)
tf.keras.backend.clear_session()
print(f"first value of gc collect: {gc.collect()}")
print(f"second value of gc collect: {gc.collect()}")
