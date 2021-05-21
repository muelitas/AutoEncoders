'''/**************************************************************************
    File: utils.py
    Author: Mario Esparza
    Date: 05/19/2021
    
    Preliminary classes and functions are saved here. Once project is more
    structured, I will move functions and classes into different files given
    their similarities.
    
***************************************************************************'''
import copy
import librosa
import os
import random
import sys

from glob import glob
from librosa.display import specshow
from librosa.feature import melspectrogram as MelSpec
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def pad_audio(audio_path, SR, longest):
    """TODO"""
    y, sr = librosa.load(audio_path, SR)
    placeholder = np.zeros(longest)
    placeholder[:y.size] = y
    return placeholder

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range"""
    max_val, min_val = matrix.max(), matrix.min()
    normalized = np.divide(np.subtract(matrix, min_val), (max_val - min_val))
    return normalized

class DataGenerator(tf.keras.utils.Sequence):
    """Generate and return data"""
    def __init__(self, paths, HP, longest, ignore):
        """Initialize generator"""
        self.paths = paths #Paths to audios
        self.dim1 = HP['dim1'] #"Frequency" Dimension
        self.dim2 = HP['dim2'] #"Time" Dimension
        self.inChannel = HP['inChannel']
        self.bs = HP['bs'] #Batch Size
        self.ignore = ignore #If last batch is incomplete, ignore it?
        self.sr = HP['sr'] #Sample Rate
        self.longest = longest #Num. of samples from longest audio in dataset
        self.mels = HP['mels']
        self.N = HP['N'] #Size of FFT Window (n_fft)
        self.HL = HP['HL'] #Hop Length

    def __len__(self):
        '''Calculate number of batches'''
        #TODO fix this calculation, depending on whether or not the last batch is ignored
        return len(self.paths) // self.bs

    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Grab audio_paths of batch
        audios_paths = self.paths[idx * self.bs : (idx+1) * self.bs]

        # Calculate Spectrograms
        X, y = self.__data_generation(audios_paths)

        return X, y

    def __data_generation(self, audios_paths):
        """
        #TODO
        """
        # Initialize X
        X = np.zeros((self.bs, self.dim1, self.dim2), dtype=np.float32)

        #load audio, pad if necessary, get spectrogram, normalize it, save it
        for idx, audio_path in enumerate(audios_paths):
            y, sr = librosa.load(audio_path, self.sr)
            #If size of audio is smaller than longest, do some padding
            if y.size < self.longest:
                y = pad_audio(audio_path, self.sr, self.longest)
        
            spec = MelSpec(y=y, sr=self.sr, n_mels=self.mels, n_fft=self.N,
                           hop_length=self.HL)
            spec = normalize_0_to_1(spec)
            X[idx] = spec
                        
        #Add inChannel (dimension needed for CNN if not incorrect)
        X = X.reshape(-1, X.shape[1], X.shape[2], self.inChannel)
        
        return X, X
    
def plot_spectrogram(spec, SR):
    '''Plot a spectrogram (frequency vs time)'''
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(spec, ref=np.max)
    img = specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

class SC_DATASET():
    """Grab paths to audios from Speech Commands (SC) Dataset. {path} is the
    location of SC in disk. {folders} are the folders inside SC that will be
    scanned. {num} is the number of files that will be grabbed from each
    folder."""
    def __init__(self, path, folders, num):
        #Initialize list that will hold paths to audios
        self.audios_paths = []
        
        #Make sure num is not greater than number of files in each folder
        for folder in folders:
            if num > len(os.listdir(path + '/' + folder)):
                print("ERROR: 'num' exceeds the number of files in this"
                      f"folder: {folder}")
                
        #Initialize list that will hold paths to audios
        self.audios_paths = []
        for folder in folders:
            self.audios_paths += glob(f"{path}/{folder}/*.wav")[:num]
            
        #Randomize audios
        random.shuffle(self.audios_paths)
        
    def __getitem__(self, n):
        """Get the 'nth' audio from the dataset"""
        return self.audios_paths[n]

    def __len__(self):
        """Return number of paths in the dataset"""
        return len(self.audios_paths)
    
    def get_longest_size(self, SR):
        '''Get size of longest audio'''
        longest = 0
        for audio_path in self.audios_paths:
            wave, _ = librosa.load(audio_path, sr=SR)
            if wave.size > longest:
                longest = wave.size
        return longest

    def check_audios(self, longest, HP):
        """Make sure that all audios' spectrograms return the same shape"""
        #We only have to worry of the "time" dimension. "Frequency" will
        #always return the value in HP['mels'].
        Dims = np.zeros(len(self.audios_paths), dtype=int)
        for idx, audio_path in enumerate(self.audios_paths):
            y, _ = librosa.load(audio_path, sr=HP['sr'])
            #If size of audio is smaller than longest, do some padding
            if y.size < longest:
                y = pad_audio(audio_path, HP['sr'], longest)
        
            #Get spectrogram and dimensions
            spec = MelSpec(y=y, sr=HP['sr'], n_mels=HP['mels'], n_fft=HP['N'], 
                           hop_length=HP['HL'])
            
            Dims[idx] = spec.shape[1]
            
        indices = (Dims != Dims[0]).nonzero()
        if not indices[0].size:
            print("You are good to go, all audios' spectrograms have the same"
                  " shape (dimensions).")
        else:
            print("ERROR: one or more spectrograms returned a different shape"
                  ". Here is the list of audio(s) that caused this issue:")
            
            for i in range(0, indices[0].size):
                print("f{indices1[0][i]}, ", end="")
            print("\nBye.")
            sys.exit()
            
        return spec.shape

def get_spectrograms(paths, HP, longest):
    '''TODO'''
    #These two values are used to mimic torchaudio.transforms.MelSpectrogram
    n = 448 #length of the FFT window, originally, it was 400
    hl = n // 2 #hop length
    
    #Get a dummy spectrogram to initialize {X}
    y, sr = librosa.load(paths[0], HP['sr'])
    spec = MelSpec(y=y, sr=sr, n_mels=HP['mels'], n_fft=n, hop_length=hl)
    dim1 = len(paths)
    dim2, dim3 = spec.shape
    X = np.zeros((dim1,dim2,dim3), dtype=np.float32)
    
    #load audio, pad if necessary, get spectrogram, normalize it, and save it
    for idx, path in enumerate(paths):
        y, sr = librosa.load(path, HP['sr'])
        #If size of audio is smaller than longest, do some padding
        if y.size < longest:
            placeholder = np.zeros(longest)
            placeholder[:y.size] = y
            y = copy.deepcopy(placeholder)
        
        spec = MelSpec(y=y, sr=sr, n_mels=HP['mels'], n_fft=n, hop_length=hl)
        spec = normalize_0_to_1(spec)
        X[idx] = spec
    
    return X

def AutoEncoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded