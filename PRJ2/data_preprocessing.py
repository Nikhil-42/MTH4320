import os
import json
import numpy as np
from scipy.io import wavfile
import itertools
from progressbar import progressbar
from matplotlib import pyplot as plt

WIDTH = 10
MAX_FREQ = 8192
SPEC_RESOLUTION = (128, 128)

def compile_examples():

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
        timeframe = sps * WIDTH

        count += (len(mix) // timeframe) * (2**solos - 1) * solos
         
    timeframe = sps * WIDTH
    dataset = np.memmap('./data/dataset.npy', dtype=mix.dtype, mode='w+', shape=(count, 2, timeframe))
    index = []

    print("Generating examples")
    
    for song in progressbar(songs):

        # Read in wav files as numpy arrays
        dirpath = song['dirpath']
        sps, mix = wavfile.read(dirpath + '/' + song['mix'])
        solos = [(solo, wavfile.read(dirpath + '/' + solo)[1]) for solo in song['solos']]
        solos.sort(key = lambda solo: np.mean(solo[1]).item())

        # Get powerset (all possible mixes and solos)
        pset = powerset(solos)

        timeframe = sps * WIDTH
        for e, group in enumerate(pset):
            if len(group) < 1: continue # skip empty set

            x = np.zeros_like(group[0][1])
            for i in range(len(group)):
                x += group[i][1]

            for j in range(len(group)):
                y = group[j][1].copy()

                # x is a mix | y is a solo

                # Slice into even chunks
                for i, t in enumerate(range(0, len(x) - timeframe, timeframe)):
                    dataset[len(index), 0] = x[t : t + timeframe]
                    dataset[len(index), 1] = y[t : t + timeframe]

                    index.append({
                        'title': song['mix'],
                        'instruments': ','.join((solo[0].split('_')[2] for solo in group)),
                        'target_instrument': group[j][0].split('_')[2],
                        'start_time': WIDTH * i,
                        'end_time': WIDTH * (i + 1),
                        'example': e,
                        'num_examples': 2**len(solos) - 1
                    })

        dataset.flush()

    print('Save metadata')
    metadata = {
        'data_dtype': str(dataset.dtype),
        'data_shape': dataset.shape,
        'samples_per_second': sps,
        'width': WIDTH,
        'total_segments': len(index),
        'index': index
    }

    with open('./data/dataset.json', 'w+') as f:
        json.dump(metadata, f)

    print('Complete')

def generate_spectrograms():
    import librosa

    with open('./data/dataset.json', 'r') as f:
        metadata = json.load(f)

    sps = metadata['samples_per_second']
    dataset = np.memmap('./data/dataset.npy', mode='r', dtype=metadata['data_dtype'], shape=tuple(metadata['data_shape']))

    n_fft = 2**int(0.5 + np.log2(2 * (sps / (MAX_FREQ / SPEC_RESOLUTION[0]) - 1)))
    hop_length = (WIDTH * sps) // SPEC_RESOLUTION[1]

    x = dataset[0, 0]
    S_x = librosa.stft((x / (np.iinfo(x.dtype).max + 1)).astype('float32'), n_fft=n_fft, hop_length=hop_length)[:SPEC_RESOLUTION[0], :SPEC_RESOLUTION[1]]

    shape = (metadata['data_shape'][0], 2, S_x.shape[0], S_x.shape[1])
    spectrograms = np.memmap('./data/spectrograms.npy', shape=shape, dtype='float32', mode='w+')


    spectrograms_metadata = {
        'spec_dtype': str(spectrograms.dtype),
        'spec_shape': spectrograms.shape,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'max_freq': MAX_FREQ,
        'spec_resolution': SPEC_RESOLUTION,
    }

    metadata.update(spectrograms_metadata)

    with open('./data/dataset.json', 'w') as f:
        json.dump(metadata, f)

    for i in progressbar(range(len(dataset))):
        x = dataset[i, 0]
        y = dataset[i, 1]
        
        S_x = librosa.stft((x / (np.iinfo(x.dtype).max + 1)).astype('float32'), n_fft=n_fft, hop_length=hop_length)[:SPEC_RESOLUTION[0], :SPEC_RESOLUTION[1]]
        S_y = librosa.stft((y / (np.iinfo(x.dtype).max + 1)).astype('float32'), n_fft=n_fft, hop_length=hop_length)[:SPEC_RESOLUTION[0], :SPEC_RESOLUTION[1]]
        spectrograms[i, 0] = np.log(np.abs(S_x) ** 2 + 1)
        spectrograms[i, 1] = np.log(np.abs(S_y) ** 2 + 1)
        
        spectrograms.flush()

def generate_subset(instrument):

    with open('./data/dataset.json', 'r') as f:
        metadata = json.load(f) 

    index = metadata['index']

    i_vals = []
    for i, clip in enumerate(index):
        if clip['target_instrument'] == instrument:
            i_vals.append(i)

    segment = np.array(i_vals)
    spectrograms = np.memmap('./data/spectrograms.npy', mode='r', dtype=metadata['spec_dtype'], shape=tuple(metadata['spec_shape']))
    subspec = spectrograms[segment]

    return subspec

if __name__ == '__main__':
    compile_examples()
    generate_spectrograms()
    generate_subset('vc')
    pass