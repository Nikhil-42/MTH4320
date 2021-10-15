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

with open('./data/spectrograms.json', 'r') as f:
    spec_metadata = json.load(f)
spectrograms = np.memmap('./data/spectrograms.npy', mode='r', dtype=spec_metadata['dtype'], shape=tuple(spec_metadata['shape']))

with open('./data/dataset.json', 'r') as f:
    metadata = json.load(f)
dataset = np.memmap('./data/dataset.npy', mode='r', dtype=metadata['dtype'], shape=tuple(metadata['shape']))

print('''
Options:
[l path] Load model for use
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
    song = index[current]
    print(f'''
        {song['title']}
                   {song['example']}/{song['num_examples']}
        <----------({current})---------->
    ''')
    x = dataset[current, 0]
    y = dataset[current, 1]
    duration = len(x) / sps
    option = input(':')
    if option[0] == 'l':
        from feedforward import *

        layers = [
            Layer(1024, True, tanh, d_tanh),
            Layer(512, True, elu, d_elu),
            Layer(1024, True, elu, d_elu),
            Layer(spectrograms.shape[-2] * spectrograms.shape[-1], False, relu, d_relu),
        ]

        model = FeedforwardNeuralNetwork(spectrograms.shape[-2] * spectrograms.shape[-1], layers)
        model.print_summary()

        model.load(option[2:])
    elif option[0] == '<':
        current -= 1 if len(option) == 1 else int(option[1:])
        continue
    elif option[0] == '>':
        current += 1 if len(option) == 1 else int(option[1:])
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
        plot_spectrogram(s_x, sps, spec_metadata['hop_length'], y_axis='linear')
        plot_spectrogram(s_y, sps, spec_metadata['hop_length'], y_axis='linear')
        import pdb; pdb.set_trace()
        if model != None:
            print('Running model')
            y_hat = model.predict(s_x.reshape(1 , s_x.shape[0] * s_x.shape[1]))
            plot_spectrogram(y_hat.reshape(s_x.shape), sps, spec_metadata['hop_length'], y_axis='linear')
        plt.show()
        continue
    elif option == 'q':
        break