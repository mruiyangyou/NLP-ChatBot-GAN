import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Config import DefaultConfig
from copy import deepcopy
import numpy as np


class CsvToDataLoader:

    def __init__(self, filepath):
        print(f'Rading {filepath}')

        df = pd.read_csv(filepath, index_col=[0], header=0)

        label_col = df.shape[1] - 1
        feature = df.iloc[:, :label_col].values
        label = df.iloc[:, label_col].values

        self.x = torch.from_numpy(feature)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# split data set
con = DefaultConfig()
dataloaders = {}
for name, path in con.data_path.items():
    data = CsvToDataLoader(path)
    dataloaders[name] = DataLoader(data, batch_size=con.batch_size)

# training time
def train(model, dataloaders, optimizer, args, criterion):
    val_max = 0
    best_model = model
    t_accus, v_accus, e_accus = [], [], []
    losses = []

    for epoch in range(1, con.epoch+1):
        t_accu, v_accu, e_accu = np.array([]), np.array([]), np.array([])
        loss_e = 0
        log = ''
        for i, batch in enumerate(dataloaders['train']):
            model.train()
            x, y = batch[0], batch[1]
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_e += loss
            accs = test(model, dataloaders, args)
            t_accu, v_accu, e_accu = np.append(t_accu, accs['train']),np.append(v_accu, accs['valid']),np.append(e_accu, accs['test'])
        avg_loss = loss_e/t_accu.shape[0]
        losses.append(round(avg_loss, 4))
        print(log.format(epoch, avg_loss, np.mean(t_accu), np.mean(v_accu), np.mean(e_accu)))
        t_accus.append(np.mean(t_accu))
        v_accus.append(np.mean(v_accu))
        e_accus.append(np.mean(e_accu))
        if val_max < np.mean(v_accu):
            best_model = deepcopy(model)

    log = "Best: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
    accs = test(best_model, dataloaders)
    print(log.format(accs['train'], accs['valid'], accs['test']))

    return best_model

def test(model, dataloaders):
    model.eval()
    accs = {}

    for mode, dataloader in dataloaders.items():
        acc = 0
        num = 0
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            pred = model(x)
            label = torch.max(pred, dim = -1)
            acc += torch.sum(label == y)
            num += label.shape[1]

        accs[mode] = acc / num

    return accs










