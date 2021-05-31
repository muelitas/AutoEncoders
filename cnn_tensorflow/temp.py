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

def pad_audio(audio_path, SR, longest):
    """Pad audio with zeros to match length of {longest}"""
    y, sr = librosa.load(audio_path, SR)
    placeholder = np.zeros(longest)
    placeholder[:y.size] = y
    return placeholder

n_fft = 448
hop_length = n_fft // 2

audio_path='/media/mario/GRAYUSB/bird/50f55535_nohash_0.wav'
y, sr = librosa.load(audio_path, sr=16000)
if y.size < 16000:
    y = pad_audio(audio_path, 16000, 16000)
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)

audio_path='/media/mario/GRAYUSB/backward/0a2b400e_nohash_1.wav'
y, sr = librosa.load(audio_path, sr=16000)
if y.size < 16000:
    y = pad_audio(audio_path, 16000, 16000)
spec2 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)

X = np.zeros((2, 128, 72), dtype=np.float32)
X[0] = spec
X[1] = spec2
X = X.reshape(-1, X.shape[1], X.shape[2], 1)

