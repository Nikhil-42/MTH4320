import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

dataset = np.memmap('./data/dataset.npy')

print('''
Options:
[<] Move to previous example
[>] Move to next example
[m] Playback mixed audio
[s] Playback solo audio
''')

current = 0
while True:
    x = dataset[current, 0]
    y = dataset[current, 1]
    
    
    