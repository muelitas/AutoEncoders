'''/**************************************************************************
    File: utils.py
    Author: Mario Esparza
    Date: 05/18/2021
    
    This is the "original" version. All of this code is working and comes from:
    https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
    and its Google Colab:
    https://colab.research.google.com/drive/1_J2MrBSvsJfOcVmYAN2-WSp36BtsFZCa#scrollTo=3RY_N3gOmfDi
 
***************************************************************************'''
import copy
import numpy as np
import torch
from torch import nn, optim

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
    
    
    h1 = hidden_n.reshape((self.n_features, self.embedding_dim))
    print(f"\tshape of h1 is: {h1.shape}")
    h2 = hidden_n.squeeze(0)
    print(f"\tshape of h2 is: {h2.shape}")
    H = h1 - h2
    print(f"\tshape of H is: {h2.shape}")
    print(f"\tsum of H is: {H.sum()}")
    

    return hidden_n.reshape((self.n_features, self.embedding_dim))

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

class RecurrentAutoencoder(nn.Module):

  def __init__(self, device, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    print(f"In RecurrentAutoencoder, at first, x.shape = {x.shape}")
    x = self.encoder(x)
    print(f"In RecurrentAutoencoder, after encoder, x.shape = {x.shape}")
    x = self.decoder(x)
    print(f"In RecurrentAutoencoder, after decoder, x.shape = {x.shape}")

    return x

def train_model(device, model, train_dataset, val_dataset, n_epochs):
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history