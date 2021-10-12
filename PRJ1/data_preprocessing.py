import os
import json
import numpy as np
from scipy.io import wavfile
import itertools
from progressbar import progressbar

width = 30

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

table = []
for song in progressbar(songs):
    # import pdb; pdb.set_trace()
    dirpath = song['dirpath']
    sps, mix = wavfile.read(dirpath + '/' + song['mix'])
    solos = [wavfile.read(dirpath + '/' + solo)[1] for solo in song['solos']]
    solos.sort(key = lambda array: np.mean(array).item())
    num_permutations = 0
    start = len(examples)
    for group in powerset(solos):
        if len(group) < 1: continue
        x = group[0]
        for i in range(1, len(group)):
            x += group[i]
        y = group[0]
        num_permutations += 1
        examples.append((x, y))
    stop = len(examples) 
    table.append((song['mix'], (start, stop)))

metadata['samples_per_second'] = sps
metadata['width'] = width

def get_song(e):
    low = 0
    high = len(table) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2
        if e >= table[mid][1][1]:
            low = mid + 1
        elif e < table[mid][1][0]:
            high = mid - 1
        else:
            return table[mid]
    return None

print(f'Subdivide examples into {width} second pieces')
timeframe = sps * width

print('Scanning dataset size and allocating disk space')
index = []

for e, (x, y) in enumerate(progressbar(examples)):
    for i, t in enumerate(range(0, len(x) - timeframe, timeframe)):
        song = get_song(e)
        index.append({
            'title': song[0],
            'start_time': width * i,
            'end_time': width * (i + 1),
            'example': e - song[1][0] + 1,
            'num_examples': song[1][1] - song[1][0]
        })

dataset = np.memmap('./data/dataset.npy', dtype=int, mode='w+', shape=(len(index), 2, timeframe))

metadata['total_segments'] = len(index)
metadata['dtype'] = str(dataset.dtype)
metadata['shape'] = dataset.shape
metadata['index'] = index

with open('./data/dataset.json', 'w+') as f:
    json.dump(metadata, f)

print('Write segments to disk')
count = 0
for x, y in progressbar(examples):
    for t in range(0, len(x) - timeframe, timeframe):
        dataset[count, 0] = x[t : t + timeframe]
        dataset[count, 1] = y[t : t + timeframe]
        count += 1
    dataset.flush()


print('Complete')
