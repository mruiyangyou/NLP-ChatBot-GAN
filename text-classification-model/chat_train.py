import torch
import torch.nn as nn
from model import Net, EmbedNet
from config import DefaultConfig
import random
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import json

opt = DefaultConfig()
all_words = []
tags = []
xy = []

with open('intents.json','r') as f:
    intents = json.load(f)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# stem and lower
ingore_words = ['?','.', '!']
all_words = [stem(word) for word in all_words if word not in ingore_words]

# remove dupicated
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print('Number of labels: ', len(tags))
# train
x_train = []
y_train  = []
for (sentence, label) in xy:
    bag = bag_of_words(sentence, all_words)
    x_train.append(bag)
    y_train.append(tags.index(label))

x_train = np.array(x_train)
y_train = np.array(y_train)
print("Shape of x_train ", x_train.shape, ' Shape of Y: ', y_train.shape )
# train loader
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

class DataSet:

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train = DataSet()
trainloader = DataLoader(train, batch_size=opt.batch_size, shuffle=True)


model = Net(opt.input_size, opt.hidden_size, opt.output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)

def train(epoches, train_loader, model):
    for epoch in range(epoches):
        loss_list = []
        avg_loss = 0
        for word, label in train_loader:
            out = model(word)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre_label = torch.argmax(out, dim=-1)
            accuracy = torch.sum(pre_label == label)/label.shape[0]
            loss_list.append(loss.item())
            avg_loss += loss.item()

        if epoch % opt.print_freq == 0:
            print('Epoch: %d Loss: %.4f Accuracy: %.4f' % (epoch, avg_loss/100, accuracy))
            avg_loss = 0

train(opt.epoches, trainloader, model)

data = {
    'model_state': model.state_dict(),
    'all_words': all_words,
    'tags': tags

}

torch.save(data, opt.load_model_path)

print( "-"*10,  "NO BUG")