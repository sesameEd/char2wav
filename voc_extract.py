#!/usr/bin/env python2
import magphase as mp
import libutils as lu
# from model import SampleRNN
import argparse
from glob import glob
from os import path
import os
import subprocess
import numpy as np
import h5py

parser = argparse.ArgumentParser(
    "read wav files under in_dir and output magphase vocoder features to out_dir"
    )
parser.add_argument('-frac', nargs='?', const=1000, default=None)
parser.add_argument('--indir', default='../blzd/wav', type=str,
                    help='default to "blzd" under parent directory')
parser.add_argument('--outdir', default='./vocoder', type=str)
parser.add_argument('--n_mag', '-M', default=60, type=int, dest='dim_mag')
parser.add_argument('--n_real', '-R', default=10, type=int, dest='dim_real')
parser.add_argument('--no_batch', action='store_false',
                    help='if activated, will not batch process files')
args = vars(parser.parse_args())
print(args)
indir = args['indir']
outdir = args['outdir']
dim_mag = args['dim_mag']
dim_real = args['dim_real']
dim_imag = dim_real
dim_f0 = 1

feats = [   'mag',       'real',  'imag',  'lf0']
dim_feats = [dim_mag, dim_real, dim_imag, dim_f0]

if not glob(outdir):
    os.mkdir(outdir)
    # subprocess.check_output('mkdir {}'.format(outdir), shell=True)

def read_binfile(filename, dim=60):
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=np.float32)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)) #.astype('float64') # This is to keep compatibility with numpy default dtype.
    return  m_data

def wav2voc(wavdir, outdir, wav_name, **kwargs):
    features = kwargs.get('features', ['mag', 'real', 'imag', 'lf0'])
    dim_feats = kwargs.get('dim_features', [60, 10, 10, 1])
    mag_dim, phase_dim = dim_feats[:2]
    mp.analysis_for_acoustic_modelling(path.join(wavdir, wav_name+'.wav'),
                                       outdir,
                                       mag_dim=mag_dim,
                                       phase_dim=phase_dim,
                                       b_const_rate=True)
    mp_all = []
    fnames_feats = ['.'.join((wav_name, f)) for f in features]
    for feat_file, dim in zip(fnames_feats, dim_feats):
        mp_all.append(read_binfile(path.join(outdir, feat_file), dim))
    with h5py.File(path.join(outdir, wav_name + '.hdf5'), "w") as f:
        f["mp_all"] = np.concatenate(mp_all, axis=1)

if __name__ == '__main__':
    b_multiproc = args['no_batch'] # False if the flag is activated, otherwise True
    wavs = glob(indir + '/*')
    if args['frac']:
        wavs = wavs[:700]
    wav_names = [path.basename(wav).split('.')[0] for wav in wavs]

    if b_multiproc:
        # quit(0)
        lu.run_multithreaded(wav2voc, indir, outdir, wav_names)
    else:
        print('True')
        for wn in wav_names:
            wav2voc(indir, outdir, wn)
