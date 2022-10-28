import torch.nn as nn
import numpy as np
import os
import torch

def cal_flatten(args):
    dim = args['conv_dim'] * ((args['embedding_dim']-(args['filter_length']-1)- (args['filter_length']-1)) // 3)
    return dim



class ArcModel(nn.Module):
    def __init__(self, args):
        super(ArcModel, self).__init__()
        self.embedding = nn.Embedding(args['word_num'], args['embedding_dim'])
        self.q_conv = nn.Conv1d(args['query_length'], args['conv_dim'], args['filter_length'])
        self.doc_conv = nn.Conv1d(args['query_length'], args['conv_dim'], args['filter_length']) # size: batch * convdim * (embedding_dim-filter+1) - batch * seq * dim * 2
        self.all_conv2d = nn.Conv2d(args['conv_dim'], args['conv_dim'], (args['filter_length'], 2)) # b * cd * (
        self.max_pooling = nn.MaxPool2d((3, 1))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(cal_flatten(args), 2)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, q, d):
        q, d = self.embedding(q), self.embedding(d)
        q, d = self.q_conv(q), self.doc_conv(d)
        q, d = self.dropout(q), self.dropout(d)

        all = torch.concat([q.unsqueeze(3), d.unsqueeze(3)], axis = 3)
        all = self.all_conv2d(all)
        all = self.max_pooling(all)
        all = self.dropout(all)
        out = self.flatten(all)
        out = self.softmax(self.linear(out))
        return out


args = {'word_num':3000, 'embedding_dim':128, 'query_length':35, 'conv_dim':35, 'filter_length':3}
model = ArcModel(args)
import random
li = [random.randint(0, 200) for _ in range(300)]
q, d = torch.tensor([random.choice(li) for _ in range(35)]).view(1, 35), torch.tensor([random.choice(li) for _ in range(35)]).view(1, 35)

print(model(q, d))
#print(cal_flatten(args))



