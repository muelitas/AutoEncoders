'''/**************************************************************************
    File: main_orig.py
    Author: Mario Esparza
    Date: 05/18/2021
    
    This is the "original" version. All of this code is working and comes from:
    https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
    and its Google Colab:
    https://colab.research.google.com/drive/1_J2MrBSvsJfOcVmYAN2-WSp36BtsFZCa#scrollTo=3RY_N3gOmfDi
    
***************************************************************************'''

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import torch

from torch import nn, optim

import torch.nn.functional as F
from arff2pandas import a2p
from utils_orig import Encoder, Decoder, RecurrentAutoencoder, train_model

def create_dataset(df):
  '''Convert: from dataframe to Pytorch Tensor'''
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

ecg_train_path = '/home/mario/Desktop/ECG5000/ECG5000_TRAIN.arff'
ecg_test_path = '/home/mario/Desktop/ECG5000/ECG5000_TEST.arff'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load data. Convert from arff to pands dataframe
with open(ecg_train_path) as f:
  train = a2p.load(f)
  
with open(ecg_test_path) as f:
  test = a2p.load(f)
  
df = train.append(test) #Add test into training
df = df.sample(frac=1.0) #Grab all data randomly

CLASS_NORMAL = 1

class_names = ['Normal','R on T','PVC','SP','UB']

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns

normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

#85% for training
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)
#10% for validation and 5% for testing
val_df, test_df = train_test_split(
  val_df,
  test_size=0.33, 
  random_state=RANDOM_SEED
)

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

model = RecurrentAutoencoder(device, seq_len, n_features, 128)
model = model.to(device)

model, history = train_model(
  device,
  model, 
  train_dataset[:10], 
  val_dataset[:2], 
  n_epochs=2
)