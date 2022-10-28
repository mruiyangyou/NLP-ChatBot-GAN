import torch.nn as nn
import numpy as np
import os
import torch


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1]
    index = index.sort(dim = dim)[0]
    return x.gather(dim, index)



class CDSSM(nn.Module):

    def __init__(self,WORD_DEPTH, CONV_DIM, FILTER_LENGTH, k, L, p):
        super(CDSSM, self).__init__()
        # query processing
        self.query_cov = nn.Conv1d(WORD_DEPTH, CONV_DIM, FILTER_LENGTH)
        nn.init.xavier_normal_(self.query_cov.weight)
        self.query_cov2 = nn.Conv1d(CONV_DIM, k, FILTER_LENGTH)
        nn.init.xavier_normal_(self.query_cov2.weight)
        self.max_pool = nn.MaxPool1d(3)


        # leanr the semantic
        self.query_linear = nn.Linear(k, L)
        nn.init.xavier_normal_(self.query_linear.weight)

        # dropout
        self.dropout = nn.Dropout(p)

        # for domucment
        self.doc_cov = nn.Conv1d(WORD_DEPTH, CONV_DIM, FILTER_LENGTH)
        nn.init.xavier_normal_(self.doc_cov.weight)
        self.doc_cov2 = nn.Conv1d(CONV_DIM, k, FILTER_LENGTH)
        nn.init.xavier_normal_(self.doc_cov2.weight)

        self.doc_linear = nn.Linear(k, L)
        nn.init.xavier_normal_(self.doc_linear.weight)
        # batch normalization
        self.q_norm = nn.BatchNorm1d(WORD_DEPTH)
        self.d_norm = nn.BatchNorm1d(WORD_DEPTH)

        # q_norm
        self.sem_q_norm = nn.BatchNorm1d(1)
        self.sem_d_norm = nn.BatchNorm1d(1)

        # concat
        self.concat_linear = nn.Linear(2*L, 2)

        self.softmax = nn.LogSoftmax()


    def forward(self,q, d):
        q, d = q.transpose(1,2), d.transpose(1,2)
        q, d = self.q_norm(q), self.q_norm(d)


        q = torch.tanh_(self.query_cov(q))
        d = torch.tanh(self.doc_cov(d))

        q = self.max_pool(q)
        d = self.max_pool(d)

        q = torch.tanh_(self.query_cov2(q))
        d = torch.tanh(self.doc_cov2(d))

        q = kmax_pooling(q, 2, 1).transpose(1,2)  # shape is B * 1 * K
        d = kmax_pooling(d, 2, 1).transpose(1,2)

        q, d = self.dropout(q), self.dropout(d)
        q, d = self.sem_q_norm(q), self.sem_d_norm(d)
        q, d = self.dropout(torch.tanh(self.query_linear(q))), self.dropout(torch.tanh(self.doc_linear(d)))

        all_doc = torch.cat([q, d], dim = 2)
        out = torch.tanh(self.concat_linear(all_doc))
        out = self.softmax(self.dropout(out))

        return out


model = CDSSM(WORD_DEPTH=3000, CONV_DIM=1024, k = 300, L = 128, FILTER_LENGTH=3, p = 0.4)
input = torch.randn(2, 20, 3000)
doc = torch.randn(2, 20, 3000)
print(model(input, doc))





