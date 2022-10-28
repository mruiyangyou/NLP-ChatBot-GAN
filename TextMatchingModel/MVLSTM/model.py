import torch
import torch.nn.functional as F
import torch.nn as nn

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1]
    index = index.sort(dim = dim)[0]
    return x.gather(dim, index)

def tensordot(a, b):
    if a.shape[-1] != a.b.shape[-1]:
        raise ValueError
    else:
        output = torch.zeros(a.shape[0], a.shape[-2], b.shape(-2))

class MVLSTM(nn.Module):

    def __init__(self, args):
        super(MVLSTM, self).__init__()
        self.embedding = nn.Embedding(args['num_words'], args['embed_dim'])
        self.bilstm = nn.LSTM(args['embed_dim'], args['hidden_size'], args['num_layers'], bidirectional = True, batch_first=True)
        self.linear = nn.Linear(args['k'], args['num_classes'])
        self.dropout = nn.Dropout(0.4)
        self.k = args['k']
        self.h, self.c = torch.zeros(args['num_layers']*2,1, args['hidden_size']), torch.zeros(args['num_layers']*2,1, args['hidden_size'])


    def forward(self, q, d):
        q, d = self.embedding(q), self.embedding(d)  # size : batch * L * embed_dim
        #x = x.unsqueeze(0)
        q, d= self.bilstm(q, (self.h, self.c))[0], self.bilstm(d, (self.h, self.c))[0]   # size : batch * L * dim
        q, d = F.normalize(q, dim = 2), F.normalize(d, dim = 2)
        out = torch.tensordot(q, d, dims = ([0,2],[0,2])).view(q.shape[0], -1)
        #out = nn.Dropout(out)
        out = kmax_pooling(out, 1, self.k)
        out = self.linear(out)





        return out



args = {'num_words':200, 'embed_dim': 64, 'hidden_size':32, 'num_layers':1, 'num_classes':2, 'query_length':10, 'k':5}
import random
model = MVLSTM(args)
li = [i for i in range(10)]
q = torch.tensor(li, dtype = torch.long).view(1,10)
random.shuffle(li)
d = torch.tensor(li, dtype = torch.long).view(1,10)
print(model(q, d))
