'''/**************************************************************************
    File: utils.py
    Author: Mario Esparza
    Date: 05/23/2021
    
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GRU, BatchNormalization
from tensorflow.keras.layers import RepeatVector, LayerNormalization, Dense, TimeDistributed
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
        self.dim1 = HP['dim1'] #"Frequency" Dimension
        self.dim2 = HP['dim2'] #"Time" Dimension
        self.inChannel = HP['inChannel']
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
                           hop_length=self.HL)
            spec = normalize_0_to_1(spec)
            X[idx] = spec
                                    
        #Add inChannel (dimension needed for CNN if not incorrect)
        X = X.reshape(-1, X.shape[1], X.shape[2], self.inChannel)
        
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
                  ". Here is the list of audio(s) that caused this issue:")
            
            for i in range(0, indices[0].size):
                print("f{indices1[0][i]}, ", end="")
            print("\nBye.")
            sys.exit()
            
        return spec.shape

def AutoEncoder(input_img):
    #ENCODER
    # input_image.shape = (2, 128, 72, 1)
    conv1 = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
    # conv1.shape = (2, 128, 72, 12)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1.shape = (2, 64, 36, 12)
    conv2 = Conv2D(24, (3, 3), activation='relu', padding='same')(pool1)
    # conv2.shape = (2, 64, 36, 24)
    BN1 = BatchNormalization()(conv2)
    # BN1.shape = (2, 64, 36, 24)
    pool2 = MaxPooling2D(pool_size=(2, 2))(BN1)
    # pool2.shape = (2, 32, 18, 24)

    #Transpose and reshape to match required input shape of GRU layers
    pool2_T = tf.transpose(pool2, perm=[0, 2, 1, 3])
    # pool2_T.shape = (2, 18, 32, 24)
    dims = pool2_T.shape
    pool2_R = tf.reshape(pool2_T, (dims[0], dims[1], dims[2] * dims[3]))
    # pool2_R.shape = (2, 18, 768)
    
    gru1 = GRU(256, activation='relu', return_sequences=True)(pool2_R)
    # gru1.shape = (2, 18, 256)
    gru2 = GRU(128, activation='relu', return_sequences=False)(gru1)
    # gru2.shape = (2, 128)
    rep_vec = RepeatVector(dims[1])(gru2)
    # rep_vec.shape = (2, 18, 128)
    LN1 = LayerNormalization()(rep_vec)
    # LN1.shape = (2, 18, 128)
    
    #DECODER
    gru3 = GRU(128, activation='relu', return_sequences=True)(LN1)
    # gru3.shape = (2, 18, 128)
    gru4 = GRU(256, activation='relu', return_sequences=True)(gru3)
    # gru4.shape = (2, 18, 256)
    time_dist = TimeDistributed(Dense(dims[2] * dims[3]))(gru4)
    # time_dist.shape = (2, 18, 768)
    LN2 = LayerNormalization()(time_dist)
    # LN2.shape = (2, 18, 768)
    
    #Reshape and transpose to match required input shape of Conv2D layers
    time_dist_R = tf.reshape(LN2, (dims[0], dims[1], dims[2], dims[3]))
    # time_dist_R.shape = (2, 18, 32, 24)
    time_dist_T = tf.transpose(time_dist_R, perm=[0, 2, 1, 3])
    # time_dist_T.shape = (2, 32, 18, 24)

    conv3 = Conv2D(24, (3, 3), activation='relu', padding='same')(time_dist_T)
    # conv3.shape = (2, 32, 18, 24)
    up1 = UpSampling2D((2,2))(conv3)
    # up1.shape = (2, 64, 36, 24)
    conv4 = Conv2D(12, (3, 3), activation='relu', padding='same')(up1)
    # conv4.shape = (2, 64, 36, 12)
    BN2 = BatchNormalization()(conv4)
    # BN2.shape = (2, 64, 36, 12)
    up2 = UpSampling2D((2,2))(BN2)
    # up2.shape = (2, 128, 72, 12)
    output_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    # output_img.shape = (2, 128, 72, 1)

    global Counter
    if Counter == 0:
        print("CNN Encoder:")
        print(f"\t-input_img.shape = {input_img.shape}")
        print(f"\t-conv1.shape = {conv1.shape}")
        print(f"\t-pool1.shape = {pool1.shape}")
        print(f"\t-conv2.shape = {conv2.shape}")
        print(f"\t-BN1.shape = {BN1.shape}")
        print(f"\t-pool2.shape = {pool2.shape}")
        print("Transpose and Reshape:")
        print(f"\t-pool2_T.shape = {pool2_T.shape}")
        print(f"\t-pool2_R.shape = {pool2_R.shape}")
        print("RNN (GRU) Encoder:")
        print(f"\t-gru1.shape = {gru1.shape}")
        print(f"\t-gru2.shape = {gru2.shape}")
        print(f"\t-rep_vec.shape = {rep_vec.shape}")
        print(f"\t-LN1.shape = {LN1.shape}")
        print("RNN (GRU) Decoder:")
        print(f"\t-gru3.shape = {gru3.shape}")
        print(f"\t-gru4.shape = {gru4.shape}")
        print(f"\t-time_dist.shape = {time_dist.shape}")
        print(f"\t-LN2.shape = {LN2.shape}")
        print("Reshape and Transpose:")
        print("Transpose and Reshape:")
        print(f"\t-time_dist_R.shape = {time_dist_R.shape}")
        print(f"\t-time_dist_T.shape = {time_dist_T.shape}")
        print("CNN Decoder:")
        print(f"\t-conv3.shape = {conv3.shape}")
        print(f"\t-up1.shape = {up1.shape}")
        print(f"\t-conv4.shape = {conv4.shape}")
        print(f"\t-BN2.shape = {BN2.shape}")
        print(f"\t-up2.shape = {up2.shape}")
        print(f"\t-output_img.shape = {output_img.shape}")
        
        Counter+=1
    
    return output_img