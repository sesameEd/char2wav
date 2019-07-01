#!/usr/bin/env python2
import magphase as mp
import libutils as lu
import argparse
from glob import glob
import os
from os import path
import numpy as np
import h5py

parser = argparse.ArgumentParser(
    "if '--mode' is 'extract' or not given, " +
    "       read wav files under 'wavdir' and output magphase vocoder features to 'vocdir'," +
    "if '--mode' is 'synth', " +
    "       synthesize wavs from the vocoder features in 'vocdir' and store in 'wavdir'"
    )
parser.add_argument('-m', '--mode', dest='mode', nargs='?', default='extract',
                    help='one of the two: \"extract\" (for extracting vocoder features from raw wav files) \
                          or \"synth\" (for synthesizing from vocoder features)')
parser.add_argument('-w', '--wavdir', default='data/normalized', type=str,
                    help='default to "data/normalized"')
parser.add_argument('-v', '--vocdir', default='data/vocoder', type=str,
                    help='default to "data/vocoder"')
parser.add_argument('-F', '--file_tkn', nargs='*', type=str, default=[])

parser.add_argument('-M', '--n_mag', default=60, type=int, dest='dim_mag')
parser.add_argument('-R', '--n_real', default=10, type=int, dest='dim_real')
parser.add_argument('-r', '--sample_rate', type=int, default=48000, dest='sample_rate')
parser.add_argument('-B', '--bit_depth', type=int, default=16)
parser.add_argument('-o', '--overwrite', action='store_true',
                    help="if turned on, existing files in vocdir (with same file \
                          tokens as those in wavdir) will be overwritten.")
parser.add_argument('--no_batch', action='store_false',
                    help='if activated, will not batch process files')
parser.add_argument('--frac', nargs='?', const=1000, default=None)
parser.add_argument('--save_space', action='store_true',
                    help='if switched on, the inter-mediate files will be deleted \
                          and only one all_vocoder.hdf5 will be preserved')
args = vars(parser.parse_args())
# print(args)
mode = args['mode']
sample_rate = args['sample_rate']
bit_depth = args['bit_depth']
dim_mag = args['dim_mag']
dim_real = args['dim_real']
dim_imag = dim_real
dim_f0 = 1
if mode == "extract":
    indir = args['wavdir']
    outdir = args['vocdir']
    in_files = glob(path.join(outdir, '*.mag'))
elif mode == "synth":
    indir = args['vocdir']
    outdir = args['wavdir']
    in_files = glob(path.join(outdir, '*.wav'))
else:
    raise ValueError("mode must be one of the two below: 'synth' or 'extract'")

if glob(outdir):
    already_in = set([path.basename(file).split('.')[0] for file in in_files])
else:
    os.mkdir(outdir)
    already_in = set()

global features, size_feats
features = ['mag', 'real', 'imag', 'lf0']
size_feats = [dim_mag, dim_real, dim_imag, dim_f0]

def read_binfile(filename, dim=60):
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=np.float32)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)) #.astype('float64') # This is to keep compatibility with numpy default dtype.
    return  m_data

def wav2voc(wavdir, outdir, wav_name): #, save_space=False, **kwargs):
    mag_dim, phase_dim = size_feats[:2]
    mp.analysis_for_acoustic_modelling(path.join(wavdir, wav_name+'.wav'),
                                       outdir,
                                       mag_dim=mag_dim,
                                       phase_dim=phase_dim,
                                       b_const_rate=True)
    mp_all = []
    fnames_feats = ['.'.join((wav_name, f)) for f in features]
    for feat_file, dim in zip(fnames_feats, size_feats):
        mp_all.append(read_binfile(path.join(outdir, feat_file), dim))
    if args['save_space']:
        for f in glob(os.path.join(outdir, wav_name+'*')):
            os.remove(f)
    with h5py.File(path.join(outdir, wav_name + '.hdf5'), "w") as f:
        f["mp"] = np.concatenate(mp_all, axis=1)

def voc_arr2wav(voc_arr, outdir, wav_name):
    mag_dim, phase_dim = size_feats[:2]

def scale_m_std(tensor_2d, ax=0):
    _m, _s = tensor_2d.mean(axis=ax), tensor_2d.std(axis=ax)
    return _m, _s, (tensor_2d - _m) / _s

def get_mean_std(tensor_2d, ax=0):
    return tensor_2d.mean(axis=ax), tensor_2d.std(axis=ax)

def concat_zip(ls, ax=0):
    return [np.concatenate(tp, axis=ax) for tp in zip(*ls)]

def mask_scale_mpl(voc_arr, lf0_dim=-1, critr=np.exp):
    if lf0_dim == -1 or lf0_dim == voc_arr.shape[1] - 1:
        _mp, _lf = np.split(voc_arr, [lf0_dim], axis=1)
    else:
        _lf = voc_arr[:, lf0_dim]
        _mp = np.delete(voc_arr, lf0_dim, 1)
    vuv = critr(_lf) > 0
    mp_ms, lf_ms = [get_mean_std(_v) for _v in (_mp, _lf[vuv].reshape(-1, 1))]
    _m, _s = concat_zip([mp_ms, lf_ms], ax=0)
    return _m, _s, vuv, (voc_arr - _m) / _s


if __name__ == '__main__':
    do_parallelize = args['no_batch']   # False if the flag is activated, otherwise True
    files = glob(indir + '/*')
    if args['frac']:
        files = files[5:10]
    all_tkn = [path.basename(file).split('.')[0] for file in files]
    if len(args['file_tkn']) > 0:
        file_tkn = args['file_tkn']
    else:
        file_tkn = all_tkn if args['overwrite'] else sorted(set(all_tkn) - already_in)

    if mode == "extract":
        if do_parallelize:
            lu.run_multithreaded(wav2voc, indir, outdir, file_tkn)
        else:
            for wn in file_tkn:
                wav2voc(indir, outdir, wn)

        all_mps = []
        for tkn in sorted(all_tkn):
            with h5py.File(os.path.join(outdir, tkn+'.hdf5'), 'r') as f:
                all_mps.append(np.array(f.get('mp')))
            if args['save_space']:
                os.remove(os.path.join(outdir, tkn+'.hdf5'))
        all_voc = np.concatenate(all_mps, axis=0)
        sent_idx = np.insert(np.cumsum([len(vec) for vec in all_mps]), 0, 0)
        assert all_voc.shape[0] == sent_idx[-1]
        if args['overwrite'] or (not glob(os.path.join(outdir, 'all_vocoder.hdf5'))):
            with h5py.File(os.path.join(outdir, '..', 'all_vocoder.hdf5'), 'w') as f:
                f['voc_utt_idx'] = sent_idx
                f['voc_mean'], f['voc_std'], voiced, all_scaled = mask_scale_mpl(all_voc)
                f['voc_scaled_cat'] = np.insert(all_scaled, -1, voiced.flatten(), axis=1)
                f['sampling_rate'] = sample_rate
                f['bit_depth'] = bit_depth
                print(sent_idx[-1], f['voc_scaled_cat'].shape,
                      f['voc_mean'][-1], f['voc_std'][-1])
        else:
            with h5py.File(os.path.join(outdir, '..', 'all_vocoder.hdf5'), 'a') as f:
                voc_dic = {k: np.array(f.pop(k)) for k in f.keys()}
                b4_uid, ending_id = np.split(voc_dic['voc_utt_idx'], [-1])
                f['voc_utt_idx'] = np.concatenate([b4_uid, sent_idx + ending_id])
                b4_unscaled = np.delete(voc_dic['voc_scaled_cat'], -2, axis=1) \
                              * voc_dic['voc_std'] + voc_dic['voc_mean']
                all_unscaled = np.concatenate([b4_unscaled, all_voc])
                f['voc_mean'], f['voc_std'], voiced, all_scaled = mask_scale_mpl(all_unscaled)
                f['voc_scaled_cat'] = np.insert(all_scaled, -1, voiced.flatten(), axis=1)
                f['sampling_rate'] = sample_rate
                f['bit_depth'] = bit_depth
                print(f['voc_utt_idx'].shape, f['voc_scaled_cat'].shape,
                      f['voc_mean'][-1], f['voc_std'][-1])

    if mode == "synth":
        # print(file_tkn)
        # print('synthesizing from dir {}, saving to dir {}'.format(indir, outdir))
        if do_parallelize:
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
        else:
            for tkn in file_tkn:
                mp.synthesis_from_acoustic_modelling(indir, tkn, outdir,
                                                      mag_dim = dim_mag,
                                                      phase_dim = dim_real,
                                                      fs = sample_rate,
                                                      fft_len=None,
                                                      pf_type='no',
                                                      b_const_rate=True)
