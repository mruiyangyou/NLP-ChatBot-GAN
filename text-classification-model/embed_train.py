import torch
from model import EmbedNet
from config import DefaultConfig
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

# load file
with open('intents.json','r') as f:
    intents = json.load(f)

file = 'checkpoints/model_0430_22:00:32.pth'
data = torch.load(file)

tags = data['tags']
all_words = data['all_words']
opt = DefaultConfig()

from collections import Counter
vocab = Counter(all_words)

xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# data set
ignore_words = ['?','.','!']
words_to_index = lambda x:[vocab[word] for word in x if word not in ignore_words]

x_train = []
y_train = []
for (sentence,label) in xy:
    x_train.append(words_to_index(sentence))
    y_train.append(tags.index(label))

# cover to tensor
x_train = np.array(x_train).reshape(26,-1)
y_train = np.array(y_train)

class Dataset:
    def __init__(self, x, y):
        self.n_samples = x.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples

train = Dataset(x_train, y_train)

from torch.utils.data import DataLoader

dataloader = DataLoader(train, batch_size=opt.batch_size, shuffle=True)

# model

model = EmbedNet(len(all_words), 64, len(tags), opt.hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)






