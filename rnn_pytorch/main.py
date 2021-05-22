'''/**************************************************************************
    File: main.py
    Author: Mario Esparza
    Date: 05/18/2021
    
    I will follow the example provided here:
    https://colab.research.google.com/drive/1_J2MrBSvsJfOcVmYAN2-WSp36BtsFZCa#scrollTo=3RY_N3gOmfDi
    and edit it so it works with spectrograms from audios. I will try with
    audios from SpeechCommands (Google) first.
    
    Note: I am padding all audios to the same size (using the longest audio as
    the size to pad to).

***************************************************************************'''
import random
import sys
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torchaudio.transforms import MelSpectrogram as MelSpec
from torch.utils.data import DataLoader as DataLoader

from utils import SC_DATASET, data_processing, get_longest_size, train_model
from utils import rnnAutoEncoder

SC_path = '/media/mario/audios/Others_Old/speech_commands_v2'
folders_of_interest = ['backward', 'bird']
num_of_files = 10 #number of files to grab from each folder
split = {'train': 0.8, 'dev': 0.1, 'test': 0.1}

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
      'sr': 16000
}

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#Determine if gpu is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Get Dataset (paths to audios)
dataset = SC_DATASET(SC_path, folders_of_interest, num_of_files)

#Get size of the audio with most samples 
longest = get_longest_size(dataset)

#Split into train, dev and test
train_set = dataset[:int(len(dataset) * split['train'])]
dev_high = len(train_set) + int(len(dataset) * split['dev'])
dev_set = dataset[len(train_set):dev_high]
test_set = dataset[-(int(len(dataset) * split['test'])):]

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}        
train_loader = DataLoader(
    dataset=train_set,
    batch_size=HP['bs'],
    shuffle=False, #Already shuffled in {SC_DATASET}
    collate_fn=lambda x: data_processing(x, longest, HP['sr'], HP['mels']),
    **kwargs
)
dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=HP['bs'],
    shuffle=False, #Already shuffled in {SC_DATASET}
    collate_fn=lambda x: data_processing(x, longest, HP['sr'], HP['mels']),
    **kwargs
)

#TODO get {seq_len} dynamically
seq_len = 81
model = rnnAutoEncoder(seq_len, HP).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=HP['e_0'])
criterion = torch.nn.L1Loss(reduction='sum').to(device)

for epoch in range(1, HP['epochs'] + 1):
    train_model(
        model, 
        device, 
        train_loader, 
        dev_loader, 
        criterion, 
        optimizer, 
        epoch
    )


'''
#Grab paths to audios
audios_paths = []
for folder in folders_of_interest:
    audios_paths += glob(f"{SC_path}/{folder}/*.wav")[:num_of_files]
    

'''
'''
#TODO Create a timer that tells how long it takes to calculate spectrograms
#Make sure to pad them

#Get spectrogram, normalize it and save it
spec = MelSpec(sample_rate=SR, n_mels=mels)(wave)
spec = normalize_0_to_1(spec)



def data_processing(data, char2int, FM=27, TM=0.125, CASE=False):
    spectrograms, labels, inp_lengths, label_lengths, filenames = [],[],[],[],[]
    
    for (spctrgrm_path, utterance) in data:   
        spec = torch.load(spctrgrm_path)
        #Apply audio transforms (frequency and time masking) to train samples
        if CASE:
            spec = tforms.FrequencyMasking(FM)(spec)
            spec = tforms.TimeMasking(int(TM * spec.shape[2]))(spec)
                                
        spec = spec.squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(chars_to_int(utterance, char2int))
        labels.append(label)
        inp_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
        filenames.append('/'.join(spctrgrm_path.strip().split('/')[-2:]))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, inp_lengths, label_lengths, filenames
'''



