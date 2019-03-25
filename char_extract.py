#!/usr/bin/python3
import argparse
from glob import glob
import os
# from scipy.io import wavfile

parser = argparse.ArgumentParser('Process and store data')
parser.add_argument("--wav_dir", default='data/wav', type=str, help='default to "data/normalized/"')
parser.add_argument('--txt_dir', default='data/txt', type=str)
parser.add_argument('-s', '--suffix', default='txt', type=str, help='suffix of script files')
parser.add_argument('--frac', action='store_true', help='processes all file under current directory')
args = parser.parse_args()

wavdir = args['wav_dir']
scriptdir = args['txt_dir']
sfx = args['suffix']
if args['frac']:
    wavs = glob(os.path.join(wavdir, '*.wav'))
else:
    wavs = glob(os.path.join(wavdir, 'A*.wav'))
wav_names = [os.path.basename(wav).split('.')[0] for wav in wavs]
