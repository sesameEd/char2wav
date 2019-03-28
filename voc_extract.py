#!/usr/bin/env python2
import magphase as mp
import libutils as lu
import argparse
from glob import glob
from os import path
import os
import numpy as np
import h5py
# from sklearn import preprocessing

parser = argparse.ArgumentParser(
    "if '--mode' is 'extract' or not given, " +
    "       read wav files under 'wavdir' and output magphase vocoder features to 'vocdir'," +
    "if '--mode' is 'synth', " +
    "       synthesize wavs from the vocoder features in 'vocdir' and store in 'wavdir'"
    )
parser.add_argument('--frac', nargs='?', const=1000, default=None)
parser.add_argument('-m', '--mode', dest='mode', nargs='?', default='extract',
                    help='one of the two: \"extract\" (for extracting vocoder features from raw wav files) \
                          or \"synth\" (for synthesizing from vocoder features)')
parser.add_argument('-w', '--wavdir', default='data/normalized', type=str,
                    help='default to "data/normalized"')
parser.add_argument('-v', '--vocdir', default='data/vocoder', type=str,
                    help='default to "data/vocoder"')
parser.add_argument('-M', '--n_mag', default=60, type=int, dest='dim_mag')
parser.add_argument('-R', '--n_real', default=10, type=int, dest='dim_real')
parser.add_argument('-r', '--sample_rate', default=48000, dest='sample_rate')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help="if turned on, existing files in vocdir (with same file \
                          tokens as those in wavdir) will be overwritten.")
parser.add_argument('--no_batch', action='store_false',
                    help='if activated, will not batch process files')
args = vars(parser.parse_args())
print(args)
mode = args['mode']
sample_rate = args['sample_rate']
dim_mag = args['dim_mag']
dim_real = args['dim_real']
dim_imag = dim_real
dim_f0 = 1
if mode == "extract":
    indir = args['wavdir']
    outdir = args['vocdir']
elif mode == "synth":
    indir = args['vocdir']
    outdir = args['wavdir']
else:
    raise ValueError("mode must be one of the two below: 'synth' or 'extract'")

if glob(outdir):
    in_files = glob(path.join(outdir, '*.hdf'))
    already_in = set([path.basename(file).split('.')[0] for file in in_files])
else:
    os.mkdir(outdir)
    already_in = set()

# feats = [   'mag',       'real',  'imag',  'lf0']
global size_feats
size_feats = [dim_mag, dim_real, dim_imag, dim_f0]

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
    dim_feats = kwargs.get('dim_features', size_feats)
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
        f["mp"] = np.concatenate(mp_all, axis=1)

if __name__ == '__main__':
    b_multiproc = args['no_batch']   # False if the flag is activated, otherwise True
    files = glob(indir + '/*')
    if args['frac']:
        files = files[:700]
    all_tkn = set([path.basename(file).split('.')[0] for file in files])
    if not args['overwrite']:
        file_tkn = all_tkn - already_in

    if mode == "extract":
        if b_multiproc:
            lu.run_multithreaded(wav2voc, indir, outdir, file_tkn)
        else:
            for wn in file_tkn:
                wav2voc(indir, outdir, wn)
        all_mps = []
        for tkn in sorted(all_tkn):
            with h5py.File(os.path.join(outdir, tkn+'.hdf5'), 'r') as f:
                all_mps.append(np.array(f.get('mp')))
        all_cat = np.concatenate(all_mps, axis=0)
        all_mean, all_std = all_cat.mean(axis=0), all_cat.std(axis=0)
        all_cat = (all_cat - all_mean) / all_std
        sent_idx = np.insert(np.cumsum([len(vec) for vec in all_mps]), 0, 0)
        with h5py.File(os.path.join(outdir, 'all_vocoder.hdf5'), 'w') as f:
            f.create_dataset('voc_scaled_cat', all_cat.shape, data=all_cat)
            f.create_dataset('voc_utt_idx', data=sent_idx, dtype=int)
            f.create_dataset('voc_mean', data=all_mean)
            f.create_dataset('voc_std', data=all_std)
            print(f['voc_scaled_cat'].shape)
            print(sent_idx[-1])

    if mode == "synth":
        lu.run_multithreaded(mp.synthesis_from_acoustic_modelling,
                             indir,
                             file_tkn,
                             outdir,
                             dim_mag,
                             dim_real,
                             sample_rate,
                             None,
                             'no',
                             True)
