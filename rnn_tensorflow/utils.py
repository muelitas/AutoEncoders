'''/**************************************************************************
    File: utils.py
    Author: Mario Esparza
    Date: 05/22/2021
    
    Preliminary classes and functions are saved here. Once project is more
    structured, I will move functions and classes into different files given
    their similarities.
    
***************************************************************************'''
import librosa
import os
import random
import sys

from glob import glob
from librosa.display import specshow
from librosa.feature import melspectrogram as MelSpec
from tensorflow.keras.layers import GRU, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.layers import LSTM, LayerNormalization

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

Counter = 0

def pad_audio(audio_path, SR, longest):
    """Pad audio with zeros to match length of {longest}"""
    y, sr = librosa.load(audio_path, SR)
    placeholder = np.zeros(longest)
    placeholder[:y.size] = y
    return placeholder

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range"""
    max_val, min_val = matrix.max(), matrix.min()
    normalized = np.divide(np.subtract(matrix, min_val), (max_val - min_val))
    return normalized

def plot_spectrogram(spec, SR, title="Mel-frequency spectrogram"):
    '''Plot a spectrogram (frequency vs time)''' 
    #Transpose, since RNN requires spec to be transposed
    spec = spec.T
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(spec, ref=np.max)
    img = specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)

class DataGenerator(tf.keras.utils.Sequence):
    """Generate and return data"""
    def __init__(self, paths, HP, longest, case='', shuffle=False):
        """Initialize generator's variables"""
        self.bs = HP['bs'] #Batch Size
        self.case = case #To differentiate between train, dev and test
        self.dim1 = HP['dim1'] #Unlike CNN, this is timesteps
        self.dim2 = HP['dim2'] #and this is n_features
        self.longest = longest #Num. of samples from longest audio in dataset
        self.mels = HP['mels']
        self.paths = paths #Paths to audios
        self.shuffle = shuffle
        self.sr = HP['sr'] #Sample Rate
        self.N = HP['N'] #Size of FFT Window (n_fft)
        self.HL = HP['HL'] #Hop Length
                        
        self.on_epoch_end()

    def __len__(self):
        '''Calculate number of batches'''
        #NOTE: If last batch is incomplete, it will be ignored (due to //)
        return len(self.paths) // self.bs

    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Grab audio_paths for batch
        audios_paths = self.paths[idx * self.bs : (idx+1) * self.bs]
        
        # Calculate Spectrograms
        X, y = self.__data_generation(audios_paths)

        return X, y

    def __data_generation(self, audios_paths):
        """Load audio, pad if necessary, get spectrogram, normalize it, and
        return it (in batches of size {self.bs})."""
        # Initialize X
        X = np.zeros((self.bs, self.dim1, self.dim2), dtype=np.float32)

        #load audio, pad if necessary, get spectrogram, normalize it, save it
        for idx, audio_path in enumerate(audios_paths):
            y, sr = librosa.load(audio_path, self.sr)
            #If size of audio is smaller than longest, do some padding
            if y.size < self.longest:
                y = pad_audio(audio_path, self.sr, self.longest)
        
            spec = MelSpec(y=y, sr=self.sr, n_mels=self.mels, n_fft=self.N,
                           hop_length=self.HL) #returns (features, timesteps)
            spec = normalize_0_to_1(spec)
            
            #Transpose to match required input shape of GRU layers
            X[idx] = spec.T
            # X[idx] = spec
        
        return X, X
    
    def on_epoch_end(self):
        """Do this at the end of each epoch (and beginning of first epoch)"""
        #Shuffle paths
        if self.shuffle == True:
            random.shuffle(self.paths)

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
                  ". Here is the list of audioe(s) that caused this issue:")
            
            for i in range(0, indices[0].size):
                print("f{indices1[0][i]}, ", end="")
            print("\nBye.")
            sys.exit()
            
        #Transpose to match required input shape of GRU layers       
        return spec.T.shape
        # return spec.shape

def AutoEncoder(inpuT, HP):
    #Using LSTMs (exactly as it is setup in one of the tutorials)
    # #Encoder
    # lstm1_enc = LSTM(128, activation='relu', return_sequences=True)(inpuT)
    # lstm2_enc = LSTM(64, activation='relu', return_sequences=False)(lstm1_enc)
    
    # rep_vec = RepeatVector(HP['dim1'])(lstm2_enc)
    # layer_norm = LayerNormalization()(rep_vec)
    
    # #Decoder
    # lstm1_dec = LSTM(64, activation='relu', return_sequences=True)(layer_norm)
    # lstm2_dec = LSTM(128, activation='relu', return_sequences=True)(lstm1_dec)
    # time_dist = TimeDistributed(Dense(HP['dim2']))(lstm2_dec)
        
    #Using GRUs (custom implementation)
    #Encoder
    gru1_enc = GRU(128, activation='relu', return_sequences=True)(inpuT)
    gru2_enc = GRU(64, activation='relu', return_sequences=False)(gru1_enc)
    
    rep_vec = RepeatVector(HP['dim1'])(gru2_enc)
    layer_norm = LayerNormalization()(rep_vec)
    
    #Decoder
    gru1_dec = GRU(64, activation='relu', return_sequences=True)(layer_norm)
    gru2_dec = GRU(128, activation='relu', return_sequences=True)(gru1_dec)
    time_dist = TimeDistributed(Dense(HP['dim2']))(gru2_dec)
    
    global Counter
    if Counter == 0:
        print(f"inpuT.shape = {inpuT.shape}")
        print(f"gru1_enc.shape = {gru1_enc.shape}")
        print(f"gru2_enc.shape = {gru2_enc.shape}")
        print(f"rep_vec.shape = {rep_vec.shape}")
        print(f"layer_norm.shape = {layer_norm.shape}")        
        print(f"gru1_dec.shape = {gru1_dec.shape}")
        print(f"gru2_dec.shape = {gru2_dec.shape}")
        print(f"time_dist.shape = {time_dist.shape}")
        
        Counter+=1
    
    return time_dist