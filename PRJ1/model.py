import numpy as np
import json
from feedforward import *
from matplotlib import pyplot as plt

with open('./data/dataset.json', 'r') as f:
    metadata = json.load(f)

dataset = np.memmap('./data/dataset.npy', mode='r', dtype=metadata['dtype'], shape=tuple(metadata['shape']))
X = dataset[:10, 0, 10]
Y = dataset[:10, 1, 10]

print('Data loaded')

layers = [
    Layer(4096, True, relu, d_relu),
    Layer(4096, True, sigmoid, d_sigmoid),
    Layer(X.shape[1], False, relu, d_relu),
]

model = FeedforwardNeuralNetwork(X.shape[1], layers)
model.print_summary()

model.fit(X, Y, X, Y, learning_rate=0.001)

fig, ax1 = plt.subplots()

# plot the losses
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# plot the accuracy
p1 = ax1.plot(Y, label = 'Data', color = 'tab:orange')
p2 = ax1.plot(model.predict(X), label = 'Prediction', color = 'tab:green')

# add a legend
ps = p1 + p2
labs = [p.get_label() for p in ps]
ax1.legend(ps, labs, loc=0)
plt.show()