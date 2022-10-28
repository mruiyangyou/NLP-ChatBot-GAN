import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
import json
import warnings

warnings.filterwarnings('ignore')
# loada tokenizrs

tokenizer_path = '/Users/marceloyou/Desktop/Pretrained_bert/bert-base-uncased-vocab.txt'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

#load text
tags = []
all_words = []
xy = []

with open('intents.json','r') as f:
    intents = json.load(f)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenizer.encode_plus(pattern,truncation=True, padding='max_length', max_length=10, return_tensors=None, return_special_tokens_mask = True,
                             return_attention_mask = True)
        #print(w)
        xy.append((w['input_ids'], tag, w['attention_mask']))

x_train = []
y_train = []
attention_mask = []

for x, y, z in xy:
    x_train.append(x)
    y_train.append(tags.index(y))
    attention_mask.append(z)

x_train, y_train, attention_mask = np.array(x_train), np.array(y_train), np.array(attention_mask)

class DataSet:

    def __init__(self, x_train, y_train, mask):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        self.mask = mask

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.mask[index]

    def __len__(self):
        return self.n_samples

train = DataSet(x_train, y_train, attention_mask)

# load dataloder
from torch.utils.data import DataLoader
trainloader = DataLoader(train, batch_size=8, shuffle=True)

#model
class BertClassfier(nn.Module):
    def __init__(self, num_classes, drpoout):
        super(BertClassfier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(drpoout)
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x, mask):
        out = self.bert(x, attention_mask = mask)
        print(out.last_hidden_state.shape)
        out = out.last_hidden_state[:,0]
        return self.softmax(self.linear(self.dropout(out)))

    def save(self, path):
        data = {'model_state': model.state_dict()}
        torch.save(data, path)

num_classes = len(tags)
model= BertClassfier(num_classes=num_classes,drpoout=0.4)

epoches = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

def train(dataloader, model, optmimizer):
    losses = []


    for i in range(1, epoches+1):
        avg_loss = 0
        correct = 0

        for batch in dataloader:
            x , y, mask = batch[0], batch[1], batch[2]
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = criterion(pred, y)
            avg_loss += loss.item()
            loss.backward()
            optmimizer.step()
            log = "Epoch: {:d}  Loss: {:.3f}  Accuracy: {:.3f}"
            label = torch.argmax(pred, dim=-1)
            correct += torch.sum(label == y)

        losses.append(avg_loss)
        acc = correct / x_train.shape[0]

        print(log.format(i, avg_loss, acc))


    return losses, model



import matplotlib.pyplot as plt
import time

def plot(losses):
    fig = plt.figure(figsize=(10,6), dpi = 300)
    plt.plot(losses)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.savefig('loss.png')


def main():
    losses, best_model = train(trainloader, model, optimizer)
    best_model.save(time.strftime('checkpoints/model_' + '%m%d_%H:%M:%S.pth'))
    plt(losses)

if __name__ == '__main__':
   main()








