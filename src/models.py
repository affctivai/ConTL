import torch.nn as nn
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
from einops.layers.torch import Reduce
import torch.nn.functional as F
import math



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConTL(nn.Module):
    def __init__(self, args):
        super(ConTL, self).__init__()

        self.args=args
        lstm_hidden_size=args.lstm_hidden_size        
        n_units=args.n_units
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )

        self.fc_layer=nn.Sequential(
            Flatten(),
            nn.Linear(4864,n_units)            
        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=n_units, nhead=2)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=1)

        rnn = nn.LSTM

        self.eeg_rnn1 = rnn(n_units, int(lstm_hidden_size), bidirectional = True)
        self.eeg_rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional = True)
        
        if args.lstm:
            fc_in_features=4*int(lstm_hidden_size)
        else:
            fc_in_features=n_units

        self.fc = nn.Linear(in_features=fc_in_features, out_features= args.n_classes)

    def convNet(self, x):
        o= self.cnn(x)

        return o

    def sLSTM(self, x):
        batch_size = x.shape[1]
    
        _, (final_h1, _) = self.eeg_rnn1(x)
        _, (final_h2, _) = self.eeg_rnn2(final_h1)

        o = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        o = self.fc(o)

        return o

    def forward(self, x):
        o=self.convNet(x)
        o=self.fc_layer(o)
        o=torch.unsqueeze(o, dim=0)
        h=self.transformer_encoder(o)
             
        if self.args.lstm:        
            o = self.sLSTM(h)
        else:
            h=torch.squeeze(h,axis=0)            
            o=self.fc(h)
        
         
        return h,o
