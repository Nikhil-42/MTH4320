import numpy as np
import json
import sys
from feedforward import *
from matplotlib import pyplot as plt

with open('./data/dataset.json', 'r') as f:
    metadata = json.load(f)

if len(sys.argv) > 0:

    from data_preprocessing import generate_subset
    instrument = sys.argv[1]
    print('Instrument: ' + instrument)
    spectrograms = generate_subset(instrument)
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2] * spectrograms.shape[3])
    
    train_split = int(len(spectrograms) * 0.6)
    test_split = int(len(spectrograms) * 0.8)
    X = spectrograms[:train_split, 0]
    Y = spectrograms[:train_split, 1]

    X_t = spectrograms[train_split:test_split, 0]
    Y_t = spectrograms[train_split:test_split, 1]
    instrument += '_'
else:
    instrument = ''
    spectrograms = np.memmap('./data/spectrograms.npy', mode='r', dtype=metadata['spec_dtype'], shape=tuple(metadata['spec_shape']))
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2] * spectrograms.shape[3])
    X = spectrograms[:1000, 0]
    Y = spectrograms[:1000, 1]

    X_t = spectrograms[1000:1100, 0]
    Y_t = spectrograms[1000:1100, 1]

print('Data loaded')

layers = [
    Layer(512, True, sigmoid, d_sigmoid),
    Layer(256, True, elu, d_elu),
    Layer(256, True, elu, d_elu),
    Layer(512, True, elu, d_elu),
    Layer(X.shape[1], True, relu, d_relu),
]

model = FeedforwardNeuralNetwork(X.shape[1], layers, l2_penalty=0.1)
model.print_summary()
print('Model constructed')
try:
    # import pdb; pdb.set_trace()
    model.fit(X, Y, X_t, Y_t, learning_rate=0.001, epochs=100, momentum=0.9)
    print('Training Complete')
except:
    print('Something went wrong')

import time
model.save(f'models/{instrument}voicer_{int(time.time())}')
