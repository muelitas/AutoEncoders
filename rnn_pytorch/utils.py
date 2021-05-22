import os
import random
import sys
import torch
import torchaudio

from glob import glob
from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram as MelSpec

import matplotlib.pyplot as plt
import torch.nn.functional as F

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range; code from
    'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
    d1, d2, d3 = matrix.size() #original dimensions
    matrix = matrix.reshape(d1, -1)
    matrix -= matrix.min(1, keepdim=True)[0]
    matrix /= matrix.max(1, keepdim=True)[0]
    matrix = matrix.reshape(d1, d2, d3)
    return matrix

def plot_spctrgrm(title, spctrgrm):
    '''Plot spctrgrm with specified {title}'''
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
    plt.show()

class SC_DATASET(Dataset):
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
    
def get_longest_size(audios_paths):
    '''Get size of longest audio'''
    longest = 0
    for audio_path in audios_paths:
        wave, _ = torchaudio.load(audio_path)
        if wave.shape[1] > longest:
            longest = wave.shape[1]
    return longest

def data_processing(data, longest, SR, mels):
    """Check audio's sample rate. Pad to {longest} if necessary. Get its
    spectrogram and normalize it. Return spectrogram and filename."""
    spectrograms, filenames = [],[]
    
    #Pad to {longest} -> get spectrogram -> normalize
    for audio_path in data:   
        wave, sr = torchaudio.load(audio_path)
        filename = audio_path.split('/')[-1].split('.')[0]
        
        #If sample rate of audio is not {SR}, notify and end program
        if sr != SR:
            print(f"ERROR: this file '{filename}' doesn't have a sample rate"
                  f" of {SR}. Check it out. Bye.")
            sys.exit()
        
        #If length of audio is not at longest, pad it with zeros
        if wave.shape[1] < longest:
            padding = (0, longest - wave.shape[1])
            wave = F.pad(wave, padding, "constant", 0)
        
        spec = MelSpec(sample_rate=sr, n_mels=mels)(wave)
        spec = normalize_0_to_1(spec)
        #By transposing here, I don't have to do so inside {rnnAutoEncoder}
        spec = spec.squeeze(0).transpose(0, 1) #Results is: [time, n_mels]
        spectrograms.append(spec)
        filenames.append(filename)
    
    #Using 'pad_sequence' to convert list to tensor (find a better way)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    return spectrograms, filenames
    
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    print(f"In Encoder, at first, x.shape = {x.shape}")
    x = x.reshape((1, self.seq_len, self.n_features))
    print(f"In Encoder, after reshape, x.shape = {x.shape}")

    x, (_, _) = self.rnn1(x)
    print(f"In Encoder, after rnn1, x.shape = {x.shape}")
    x, (hidden_n, _) = self.rnn2(x)
    print(f"In Encoder, after rnn2, x.shape = {x.shape}")
    print(f"In Encoder, after rnn2, hidden_n.shape = {hidden_n.shape}")

    # return hidden_n.reshape((self.n_features, self.embedding_dim))
    return hidden_n.squeeze(0)

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    print(f"In Decoder, at first, x.shape = {x.shape}")
    x = x.repeat(self.seq_len, self.n_features)
    print(f"In Decoder, after repeat, x.shape = {x.shape}")
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    print(f"In Decoder, after reshape, x.shape = {x.shape}")

    x, (hidden_n, cell_n) = self.rnn1(x)
    print(f"In Decoder, after rnn1, x.shape = {x.shape}")
    x, (hidden_n, cell_n) = self.rnn2(x)
    print(f"In Decoder, after rnn2, x.shape = {x.shape}")
    x = x.reshape((self.seq_len, self.hidden_dim))
    print(f"In Decoder, after reshape, x.shape = {x.shape}")

    return self.output_layer(x)

class rnnAutoEncoder(nn.Module):

  def __init__(self, seq_len, HP):
    super(rnnAutoEncoder, self).__init__()
    n_features = HP['mels']
    embedding_dim = HP['gru_hid_dim']

    self.encoder = Encoder(seq_len, n_features, embedding_dim)
    self.decoder = Decoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    print(f"In RecurrentAutoencoder, at first, x.shape = {x.shape}")
    x = self.encoder(x)
    print(f"In RecurrentAutoencoder, after encoder, x.shape = {x.shape}")
    x = self.decoder(x)
    print(f"In RecurrentAutoencoder, after decoder, x.shape = {x.shape}")

    return x

def train_model(model, device, train_loader, dev_loader, criterion, optimizer,
                epoch):
    history = {'train': [], 'dev': []}
    model.train()
    
    train_losses = []
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, filenames = _data
        spectrograms = spectrograms.to(device)

        optimizer.zero_grad()
        
        output = model(spectrograms)
        
        loss = criterion(output, spectrograms)
        train_losses.append(loss.detach().item())
        loss.backward()
        
        optimizer.step()
        
    dev_losses = []
    model = model.eval()
    
    with torch.no_grad():
        for batch_idx, _data in enumerate(dev_loader):
            spectrograms, filenames = _data 
            spectrograms = spectrograms.to(device)
            
            output = model(spectrograms)
            
            loss = criterion(output, spectrograms)
            dev_losses.append(loss.detach().item())
            
    #Epoch's average losses 
    train_loss = sum(train_losses) / len(train_losses) #epoch's average loss
    dev_loss = sum(dev_losses) / len(dev_losses) #epoch's average loss
    history['train'].append(train_loss)
    history['dev'].append(dev_loss)
    
    return history
    