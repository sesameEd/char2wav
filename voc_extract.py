#!/usr/bin/python2
import magphase as mp
# import libutils as lu
# from model import SampleRNN
import argparse
from glob import glob
from os import path
import subprocess
import numpy as np
import h5py

parser = argparse.ArgumentParser(
    "read wav files under in_dir and output magphase vocoder features to out_dir"
    )
parser.add_argument('--indir', default='../blzd/wav', type=str,
                    help='default to "blzd" under parent directory')
parser.add_argument('--outdir', default='./vocoder', type=str)
parser.add_argument('--n_mag', '-M', default=60, type=int, dest='dim_mag')
parser.add_argument('--n_real', '-R', default=10, type=int, dest='dim_real')
# parser.add_argument('--n_imag', '-I', default=10, type=int)
args = vars(parser.parse_args())

indir = args['indir']
outdir = args['outdir']
dim_mag = args['dim_mag']
dim_real = args['dim_real']
dim_imag = args['dim_real']
dim_f0 = 1

feats = [   'mag',       'real',  'imag',  'lf0']
dim_feats = [dim_mag, dim_real, dim_imag, dim_f0]
print(dim_feats)

if not glob(outdir):
    subprocess.check_output('mkdir {}'.format(outdir), shell=True)

def read_binfile(filename, dim=60):
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=np.float32)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    # m_data = np.squeeze(m_data)
    return  m_data

def wav2voc(wavdir, outdir, features):
    wavs = glob(wavdir + '/*')
    # print wavs[1]
    wav_names = [path.basename(wav).split('.')[0] for wav in wavs]
    for wn in wav_names:
        mp.analysis_for_acoustic_modelling(path.join(wavdir, wn+'.wav'),
                                           outdir,
                                           mag_dim=60,
                                           phase_dim=10,
                                           b_const_rate=True)
        mp_features = []
        fnames_feats = ['.'.join((wn, f)) for f in feats]
        for feat_file, dim in zip(fnames_feats, dim_feats):
            mp_features.append(read_binfile(path.join(outdir, feat_file), dim))
        with h5py.File(path.join(outdir, wn + '.hdf5'), "w") as f:
            f["cat_mp"] = np.concatenate(mp_features, axis=1)

if __name__ == '__main__':
    wav2voc(indir, outdir, feats)
