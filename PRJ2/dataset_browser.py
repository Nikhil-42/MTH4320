import time
import json
import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

import librosa
from librosa.display import specshow

def plot_spectrogram(Y, sr, hop_length, y_axis='linear'):
    plt.figure(figsize=(25,10))
    g = specshow(Y, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format="%+2.f")

with open('./data/dataset.json', 'r') as f:
    metadata = json.load(f)
dataset = np.memmap('./data/dataset.npy', mode='r', dtype=metadata['data_dtype'], shape=tuple(metadata['data_shape']))
spectrograms = np.memmap('./data/spectrograms.npy', mode='r', dtype=metadata['spec_dtype'], shape=tuple(metadata['spec_shape']))

print('''
Options:
[l filename] Load model for use
[<#] Move to (#) previous example
[>#] Move to (#) next example
[m] Playback mixed audio
[s] Playback solo audio
[sp] Print spectrogram
[p] Print waveforms
[q] Quit
''')

sps = metadata['samples_per_second']
index = metadata['index']

current = 0
model = None
while True:
    try:
        current = (current + len(index)) % len(index)
        song = index[current]
        print(
            f'{song["title"]}'.center(100) + '\n' +
            f'{song["target_instrument"]} | {song["instruments"]}'.center(100) + '\n' +
            f'{song["start_time"] // 60 :02}:{song["start_time"] % 60 :02} - {song["end_time"] // 60 :02}:{song["end_time"] % 60 :02}'.center(100) + '\n' +
            f'{song["example"]}/{song["num_examples"]}'.center(100) + '\n' +
            f'<----------({current})---------->'.center(100) + '\n'
        )
        x = dataset[current, 0]
        y = dataset[current, 1]
        duration = len(x) / sps
        option = input(':')
        if option[0] == 'l':
            from convolutional import *
            from torchsummary import summary
            import torch

            path = os.path.join('saved_models/MUSIC', option[2:])
            model = torch.load(path)
            model.to('cuda')
            model.eval()

            summary(model, input_size=(1, 128, 128))

        elif option[0] == '<' or option[0] == '>':
            mul = 1 if option[0] == '>' else -1
            p = option.split(' ')
            command = p[0]
            steps = 1 if len(p) == 1 else int(p[1])
            if len(command) == 1 :
                current += mul * steps
                current %= len(index)
            elif len(command) == 2:
                for _ in range(steps):
                    current_ex = index[current]['example']
                    while index[current]['example'] == current_ex:
                        current += mul
                        current %= len(index)
            elif len(command) == 3:
                for _ in range(steps):
                    current_title = index[current]['title']
                    while index[current]['title'] == current_title:
                        current += mul
                        current %= len(index)
            else:
                current = 0 if mul == -1 else len(index) - 1
            continue
        elif option == 'm':
            sd.play(x.astype('int32'), sps)
            continue
        elif option == 's':
            sd.play(y.astype('int32'), sps)
            continue
        elif option == 'p':
            plt.plot(x, color='blue')
            plt.plot(y, color='orange')
            plt.ylim([np.iinfo(x.dtype).min,np.iinfo(x.dtype).max])
            
            plt.show()
            continue
        elif option == 'sp':
            s_x = spectrograms[current, 0]
            s_y = spectrograms[current, 1]
            plot_spectrogram(s_x, sps, metadata['hop_length'], y_axis='linear')
            plot_spectrogram(s_y, sps, metadata['hop_length'], y_axis='linear')
            
            if model != None:
                print('Running model')
                y_hat = model(torch.tensor(s_x[None, None, :]).to('cuda')).cpu().detach().numpy()
                plot_spectrogram(y_hat.reshape(s_x.shape), sps, metadata['hop_length'], y_axis='linear')
            plt.show()
            continue
        elif option == 'q':
            break
    except Exception as e:
        print(e)