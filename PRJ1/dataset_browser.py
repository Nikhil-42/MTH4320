import time
import json
import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

with open('./data/dataset.json', 'r') as f:
    metadata = json.load(f)
    dataset = np.memmap('./data/dataset.npy', dtype=metadata['dtype'], shape=tuple(metadata['shape']))

print('''
Options:
[<] Move to previous example
[>] Move to next example
[m] Playback mixed audio
[s] Playback solo audio
[p] Print waveforms
[q] Quit
''')

sps = metadata['samples_per_second']
index = metadata['index']

current = 0
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
    if option == '<':
        current -= 1
        continue
    elif option == '>':
        current += 1
        continue
    elif option == 'm':
        sd.play(x.astype('int32'), sps)
        time.sleep(duration)
        sd.stop()
        continue
    elif option == 's':
        sd.play(y.astype('int32'), sps)
        time.sleep(duration)
        sd.stop()
        continue
    elif option == 'p':
        plt.plot(x, color='blue')
        plt.plot(y, color='orange')
        plt.ylim([-1000000000,1000000000])
        
        plt.show()
        continue
    elif option == 'q':
        break