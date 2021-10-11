import os
import json
import numpy as np
from scipy.io import wavfile
import itertools
from progressbar import progressbar

print('Reading file tree')
wav_files = list()
for dirpath, dirnames, filenames in os.walk('./data/', topdown=False):
    wav_files += [(dirpath, filename) for filename in filenames if filename[-3:] == 'wav' and filename[0] != '.']

songs = []
for dirpath, filename in wav_files:
    if filename[:5] == 'AuMix':
        songs.append({'dirpath': dirpath, 'mix': filename, 'solos': list()})
    elif dirpath == songs[-1]['dirpath']:
        songs[-1]['solos'].append(filename)
    else:
        raise Exception('%s is out of sequence' % filename)

print('Generating examples')
examples = list()
metadata = {}

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

index = {}
for song in progressbar(songs):
    start = len(examples)
    dirpath = song['dirpath']
    sps, mix = wavfile.read(dirpath + '/' + song['mix'])
    solos = [wavfile.read(dirpath + '/' + solo)[1] for solo in song['solos']]
    solos.sort(key = lambda array: np.mean(array).item())
    for group in powerset(solos):
        if len(group) < 1: continue
        x = group[0]
        for i in range(1, len(group)):
            x += group[i]
        y = group[0]
        examples.append((x, y))
    stop = len(examples) - 1
    index[song['mix']] = (start, stop)

width = 30

metadata['index'] = index
metadata['samples_per_second'] = sps
metadata['width'] = width


print(f'Subdivide examples into {width} second pieces')
timeframe = sps * width

print('Scanning dataset size and allocating disk space')
count = 0
start = 0
for i, (x, y) in progressbar(enumerate(examples)):
    for t in range(0, len(x) - timeframe, timeframe):
        count += 1

dataset = np.memmap('./data/dataset.npy', dtype=int, mode='r+', shape=(count, 2, timeframe))

metadata['total_segments'] = count
metadata['dtype'] = str(dataset.dtype)
metadata['shape'] = dataset.shape

with open('./data/dataset.json', 'w+') as f:
    json.dump(metadata, f)

exit()
print('Write segments to disk')
count = 0
for x, y in progressbar(examples):
    for t in range(0, len(x) - timeframe, timeframe):
        dataset[count, 0] = x[t : t + timeframe]
        dataset[count, 1] = y[t : t + timeframe]
        count += 1
    dataset.flush()


print('Complete')
