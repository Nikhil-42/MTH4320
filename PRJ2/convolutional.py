import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary

from datetime import datetime
import os
from data_preprocessing import generate_subset

import numpy as np
from matplotlib import pyplot as plt
from utils import *

from torch.utils.tensorboard import SummaryWriter
from progressbar import progressbar

class MusicModel(nn.Module):

    def __init__(self):
        super(MusicModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, 2, padding=2)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')
        self.elu2 = nn.ELU()

        self.resonator = nn.Conv2d(16, 1, (2, 16), (2, 2))
        self.res_lin = nn.Linear(25*32, 64)
        self.elu3 = nn.ELU()

        self.conv_transpose = nn.ConvTranspose2d(16, 1, 5, 2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = self.conv1(x)
        a = self.elu1(a)
        a = self.conv2(a)
        a = self.elu2(a)

        resonance = self.resonator(a).view(a.shape[0], -1)
        timeline = self.res_lin(resonance).view(a.shape[0], 1, -1, 1)
        timeline = self.elu3(timeline)

        a = a * timeline
        a = self.conv_transpose(a)[:, :, :-1, :-1]
        a = self.relu(a)

        return a

class SpectrogramDataset(Dataset):

    def __init__(self, split='train', instrument='vn'):
        self.data = generate_subset(instrument)
        if split == 'train':
            self.data = self.data[:(len(self.data) * 6) // 10]
        elif split == 'test':
            self.data = self.data[(len(self.data) * 6) // 10:(len(self.data) * 8) // 10]
        elif split == 'dev':
            self.data = self.data[(len(self.data) * 8) // 10:]
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return [torch.tensor(self.data[index, 0:1]), torch.tensor(self.data[index, 1:])] 


def train(model, train_ds, epochs=100, batch_size=32, criterion=nn.BCELoss(), optimizer=None, device='cpu', writer=None):
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)

    model.train()
    model.to(device)

    print("Beginning training.")
    loss = 'nan'

    for epoch in progressbar(range(0, epochs), min_value=0, max_value=epochs):
        running_loss = 0
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = criterion(y_pred, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # import pdb; pdb.set_trace() # uncomment to interact with model while training

            for name, param in model.named_parameters():
                writer.add_histogram("Live/"+name, param)

        writer.add_scalar('Live/Loss', running_loss, epoch)
            

    print("Training complete.")

def validate(model, test_ds, precision=0.75, criterion=nn.BCELoss(), device='cpu', writer=None):
    model.eval()
    model.to(device)

    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False)
    sum_loss = 0

    with torch.no_grad():
        correct = 0
        print("Beginning evaluation.")
        for i, (inputs, targets) in progressbar(enumerate(test_dl), 0, len(test_dl)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            y_pred = model(inputs)
            sum_loss += criterion(y_pred, targets)
            
            correct += (torch.sum(y_pred.norm() * targets.norm()).item() > precision)
        print("Evaluation complete.")
        return correct/len(test_ds), sum_loss/len(test_ds)

if __name__ == "__main__":
    try:
        while True:
            if bool_input('Load model from disk? (y/N): ', default=False):
                path = os.path.join('saved_models/MUSIC', str_input('filename: '))
                model = torch.load(path)
            else:
                model = MusicModel()
            
            model.to('cuda')
            summary(model, input_size=(1, 128, 128))

            device = str_input("device (cuda): ")
            device = 'cuda' if len(device) == 0 else device
            model.to(device)

            instrument = str_input('instrument: ')

            last_run = sorted([dirname for dirname in next(os.walk('runs'))[1] if instrument in dirname])
            last_run = last_run[-1] if len(last_run) > 0 else f'MUSIC.{instrument}_-1'
            exp_num = int(last_run[last_run.index('_') + 1:]) + 1

            writer = SummaryWriter(f'runs/MUSIC.{instrument}_{exp_num}')
            
            if bool_input("Perform additional training? (Y/n): "):
                train_ds = SpectrogramDataset(split='train', instrument=instrument)

                train_kwargs = {}
                # if bool_input("Resume from checkpoint? (y/N): ", default=False):
                #     train_kwargs['checkpoint'] = torch.load(str_input("path: "))
                # else:
                epochs_str = str_input("epochs (100): ")
                batch_size_str = str_input("batch_size (32): ")
            
                train_kwargs['epochs'] = 100 if len(epochs_str) == 0 else int(epochs_str)
                train_kwargs['batch_size'] = 32 if len(batch_size_str) == 0 else int(batch_size_str)
                train_kwargs['device'] = device
        
                train_kwargs['writer'] = writer
                train_kwargs['criterion'] = nn.MSELoss()

                train_kwargs['optimizer'] = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

                train(model, train_ds, **train_kwargs)

                strtime = datetime.now().strftime('%S.%M.%H-%d.%m.%Y')
                os.makedirs('saved_models/{}/'.format("MUSIC"), exist_ok=True)
                torch.save(model, 'saved_models/{}/{}_{}.pt'.format("MUSIC", instrument, strtime))
            
            if bool_input("Perform evaluation of model? (Y/n): "):
                precision = float_input('precision (0.75):')

                test_ds = SpectrogramDataset(split='test')
                accuracy, loss = validate(model, test_ds, writer=writer, criterion=nn.MSELoss(), device=device)
                print('Accuracy: ', accuracy)
                print('Loss: ', loss)

            if bool_input("Add to TensorBoard? (y/N): ", False):
                test_ds = SpectrogramDataset(split='test')
                writer.add_graph(model, test_ds[0][0].unsqueeze(0).to(device))
            
            writer.close()
            print("Going back to head.")

    except KeyboardInterrupt:
        print("Exiting")

        