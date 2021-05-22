'''/**************************************************************************
    File: temp.py
    Author: Mario Esparza
    Date: 05/19/2021
    
    Before implementing in main, I use this file to run dummy tests.
    
***************************************************************************'''
import librosa
import numpy as np

from librosa.display import specshow
from utils import normalize_0_to_1

audio_path='/media/mario/GRAYUSB/bird/50f55535_nohash_0.wav'
audio, sr = librosa.load(audio_path, sr=16000)
n_fft = 448
hop_length = n_fft // 2
spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

S_dB = librosa.power_to_db(spec, ref=np.max)

img = specshow(S_dB, x_axis='time',

                         y_axis='mel', sr=sr,  ax=ax)

fig.colorbar(img, ax=ax, format='%+2.0f dB')

ax.set(title='Mel-frequency spectrogram')

spec2 = normalize_0_to_1(spec)
fig, ax = plt.subplots()

S_dB = librosa.power_to_db(spec2, ref=np.max)

img = specshow(S_dB, x_axis='time',

                         y_axis='mel', sr=sr,

                         fmax=8000, ax=ax)

fig.colorbar(img, ax=ax, format='%+2.0f dB')

ax.set(title='Mel-frequency spectrogram')
