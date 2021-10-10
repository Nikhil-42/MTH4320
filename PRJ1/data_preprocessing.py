import os
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

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

for song in progressbar(songs):
    dirpath = song['dirpath']
    mix = wavfile.read(dirpath + '/' + song['mix'])[1]
    # import pdb; pdb.set_trace()
    solos = [wavfile.read(dirpath + '/' + solo)[1] for solo in song['solos']]
    solos.sort(key = lambda array: np.mean(array).item())
    for group in powerset(solos):
        if len(group) < 1: continue
        x = group[0]
        for i in range(1, len(group)):
            x += group[i]
        y = group[0]
        examples.append((x, y))

width = 30
print(f'Subdivide examples into {width} second pieces')
timeframe = 48000 * width
import pdb; pdb.set_trace()

print('Scanning dataset size and allocating disk space')
count = 0
for x, y in progressbar(examples):
    for i in range(0, len(x) - timeframe, timeframe):
        count += 1
dataset = np.memmap('./data/dataset.npy', dtype=int, mode='w+', shape=(count, 2, timeframe))

print('Write segments to disk')
count = 0
for x, y in progressbar(examples):
    for i in range(0, len(x) - timeframe, timeframe):
        dataset[count, 0] = x[i : i + timeframe]
        dataset[count, 1] = y[i : i + timeframe]
        count += 1
    dataset.flush()

print('Complete')
