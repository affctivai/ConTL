import torch.nn as nn
import torch


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
        o=self.transformer_encoder(o)
             
        if self.args.lstm:        
            o = self.sLSTM(o)
        else:
            o=torch.squeeze(o,axis=0)            
            o=self.fc(o)
        
         
        return o

'''Baseline models'''

class MT_CNN(nn.Module):
    def __init__(self, args):
        super(MT_CNN, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            Flatten(),
            nn.Linear(9920,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )

        self.fc = nn.Linear(in_features=512, out_features= args.n_classes)

    def forward(self, x):
        x=self.cnn_layer(x)
        x=self.fc(x)

        return x


#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class DGCNN(nn.Module):
    def __init__(self, args):
        #in_channels(int): The feature dimension of each electrode.

        if args.data_choice =='dreamer':
            in_channels=3
            num_electrodes=14
        elif args.data_choice == 'deap':    
            in_channels = 4
            num_electrodes = 32
        else:
            in_channels= 5
            num_electrodes=62
                
        k_adj = 2
        num_classes=args.n_classes
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        out_channels=256
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        x=torch.squeeze(x, axis=1)

        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)

        return result


