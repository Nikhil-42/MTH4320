import os
import json
import numpy as np
from scipy.io import wavfile
import itertools
from progressbar import progressbar

width = 10

print('Reading file tree')
wav_files = []
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

print('Allocating storage space')

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

count = 0

sps = 0
# Dry run to get count
for song in progressbar(songs):

    # Read in wav files as numpy arrays
    dirpath = song['dirpath']
    sps, mix = wavfile.read(dirpath + '/' + song['mix'])
    solos = len(song['solos'])
    timeframe = sps * width

    count += (len(mix) // timeframe) * (2**solos - 1)

timeframe = sps * width
dataset = np.memmap('./data/dataset.npy', dtype='float64', mode='w+', shape=(count, 2, timeframe))
metadata = {}
index = []

print('Generating examples')

for song in progressbar(songs):

    # Read in wav files as numpy arrays
    dirpath = song['dirpath']
    sps, mix = wavfile.read(dirpath + '/' + song['mix'])
    solos = [(solo, wavfile.read(dirpath + '/' + solo)[1]) for solo in song['solos']]
    solos.sort(key = lambda solo: np.mean(solo[1]).item())

    # Get powerset (all possible mixes and solos)
    pset = powerset(solos)

    timeframe = sps * width
    for e, group in enumerate(pset):
        if len(group) < 1: continue # skip empty set

        x = np.zeros_like(group[0][1])
        for i in range(len(group)):
            x += group[i][1]
        y = group[0][1].copy()

        # x is a mix | y is a solo

        # Slice into even chunks
        for i, t in enumerate(range(0, len(x) - timeframe, timeframe)):
            dataset[len(index), 0] = x[t : t + timeframe]
            dataset[len(index), 1] = y[t : t + timeframe]

            index.append({
                'title': song['mix'],
                'instruments': ','.join((solo[0] for solo in group)),
                'start_time': width * i,
                'end_time': width * (i + 1),
                'example': e,
                'num_examples': 2**len(solos) - 1
            })

    dataset.flush()

print('Save metadata')
metadata['samples_per_second'] = sps
metadata['width'] = width
metadata['total_segments'] = len(index)
metadata['dtype'] = str(dataset.dtype)
metadata['shape'] = dataset.shape
metadata['index'] = index

with open('./data/dataset.json', 'w+') as f:
    json.dump(metadata, f)

print('Complete')
