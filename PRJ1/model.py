import numpy as np
import json
from feedforward import *
from matplotlib import pyplot as plt

with open('./data/spectrograms.json', 'r') as f:
    metadata = json.load(f)

spectrograms = np.memmap('./data/spectrograms.npy', mode='r', dtype=metadata['dtype'], shape=tuple(metadata['shape']))
spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2] * spectrograms.shape[3])
X = spectrograms[:1000, 0]
Y = spectrograms[:1000, 1]

X_t = spectrograms[1000:1100, 0]
Y_t = spectrograms[1000:1100, 1]

# import pdb; pdb.set_trace()

print('Data loaded')

layers = [
    Layer(512, True, tanh, d_tanh),
    Layer(256, True, elu, d_elu),
    Layer(256, True, elu, d_elu),
    Layer(256, True, elu, d_elu),
    Layer(512, True, elu, d_elu),
    Layer(X.shape[1], False, relu, d_relu),
]

model = FeedforwardNeuralNetwork(X.shape[1], layers)
model.print_summary()
print('Model constructed')
try:
    import pdb; pdb.set_trace()
    model.fit(X, Y, X_t, Y_t, learning_rate=0.01, epochs=100, momentum=0.75)
    print('Training Complete')
except:
    print('Something went wrong')

import time
model.save(f'models/voicer_{int(time.time())}')
