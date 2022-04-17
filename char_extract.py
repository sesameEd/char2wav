#!/usr/bin/env python3
import argparse
import re
import h5py
from operator import itemgetter
import numpy as np
import os
from glob import glob

parser = argparse.ArgumentParser('Process transcripts and convert to index-encode array')
parser.add_argument('-a', '--alltxtpath', default='data/title/all.txt', type=str,
                    help=('the file that stores all transcripts with utterances separated by \\n. '
                          'Tokenized transcript is assumed to be in the same dir with "tknzd_" prefix. '
                          'Output .hdf5 file will be stored in its parent directory if not specified. '))
parser.add_argument('-d', '--dict', default='data/title/vocab.freq', type=str,
                    help='the tsv file storing all character types (with counts)')
parser.add_argument('-t', '--titles', default='data/title/title.freq', type=str,
                    help='the tsv file storing the different titles (with counts)')
parser.add_argument('-o', '--outdir', type=str, help='the directory to save output .hdf5 file, defaults to ./data')
args = vars(parser.parse_args())
do_align = True
boundary_signs = dict(zip(['BOS', 'BOW', 'EOS', 'EOW'], map(str, range(1, 5))))
BOS, BOW, EOS, EOW = map(str, range(1, 5))
bndry_types = np.string_(['BOS', 'BOW', 'EOS', 'EOW'])

vocab_file = args['dict']
title_file = args['titles']
script_file = args['alltxtpath']
dir_name, file_name = os.path.split(script_file)
tknzd_file = os.path.join(dir_name, 'tknzd_' + file_name)
outdir = args.get('outdir')
if not outdir:
    outdir = os.path.dirname(dir_name)

def reverse_dic(input_dic):
    return {v: k for k, v in input_dic.items()}

def get_begin_ids(list_sents):
    """
    returns an array of beginning indices of sentences (with spaces removed);
    an extra index of the length of the entire concatenated sentence array is added;
    s.t. the individual sents can be indexed by concat_array[id[i]:id[i+1]]
    """
    lens = np.array([len(line) - line.count(' ') for line in list_sents])
    return np.cumsum(np.insert(lens, 0, 0), 0)

def get_bow_eow(list_sents):
    lens = np.array([len(word) for sent in list_sents for word in sent.split(' ')])
    lens = np.cumsum(np.insert(lens, 0, 0), 0)
    bow = np.delete(lens, -1, 0)
    eow = np.delete(lens-1, 0, 0)
    return bow, eow

def upper_or_lower(char):
    return 1 if char.isupper() else 0

def bin_annotate_1d(indices, len_of_array):
    arr = np.zeros(len_of_array)
    np.put(arr, indices, 1)
    return arr

def shift_indices(original_indices, labelled_arr):
    shift, shifted_ids = 0, [0]
    for _i, _is in zip(original_indices[:-1], original_indices[1:]):
        shift += np.sum(labelled_arr[_i:_is])
        shifted_ids.append(_is + shift)
    return np.array(shifted_ids)

def arr2sent(encoded_arr, i2c_dic, i2bnd_dic=None, up_indctr=None):
    """
    :i2c_dic the dictionary mapping num encoding to characters
    :i2bnd_dic the dictionary mapping num encodings to EOS, EOW, etc., signs (sentence/word segmentation)
    :up_indctr the array where upper case letters are tagged '1' and otherwise (0 or 2) elsewhere
    """
    if up_indctr is not None:
        _ids = np.where(up_indctr == 1)[0]
        _sent = ''.join(idx2char[c].upper() if i in _ids else idx2char[c]
                        for i, c in enumerate(encoded_arr))
    else:
        _sent = ''.join([i2c_dic[_] for _ in encoded_arr])
    if i2bnd_dic:
        BOS, BOW, EOS, EOW = itemgetter('BOS', 'BOW', 'EOS', 'EOW')(i2bnd_dic)
        return _sent.replace(BOS + BOW, '').replace(EOW+EOS, '\n').replace(EOW+BOW, ' ',)
    else:
        return _sent


if __name__ == "__main__":
    with open(script_file, 'r') as f:
        all_scripts = [line.rstrip() for line in f]
    utt_id = get_begin_ids(all_scripts)

    # get the indices of word and sentence boundaries, for now all are indexed by space-removed text
    if not os.path.isfile(tknzd_file):
        print("Cannot find tokenized scripts in {0}. Assuming {1} tokenized.".format(
            tknzd_file, script_file))
        tknzd_file = script_file
        do_align = False
    with open(tknzd_file, 'r') as f:
        all_sents = [line.rstrip() for line in f]
    bos_id = get_begin_ids(all_sents)
    eos_id = np.delete(bos_id-1, 0, 0)
    bos_id = np.delete(bos_id, -1, 0)
    bow_id, eow_id = get_bow_eow(all_sents)
    assert eos_id[-1] == eow_id[-1], 'did not get the right sentence or word boundaries'
    assert eos_id[-1] + 1 == utt_id[-1], ('index of last character does not accord',
                                          eos_id[-1] + 1, utt_id[-1])
    assert all(np.in1d(utt_id[:-1], bow_id, assume_unique=True)), \
        'utts no. {} didn\'t start with word boundary.'.format(
            np.where(np.in1d(utt_id[:-1], bow_id))[0])
    if do_align:
        ebows_inplace = np.sum([bin_annotate_1d(ids, utt_id[-1])
                                for ids in [bos_id, eos_id, bow_id, eow_id]], 0)
        assert ebows_inplace.shape == (utt_id[-1],), 'Error with annotating 1d-array'
        utt_id = shift_indices(utt_id, ebows_inplace)
    # get an invertible mapping between individual characters (including boundaries) and numeric indices
    with open(vocab_file, 'r') as f:
        chars = [re.sub(r'^\d+\t', '', kv).rstrip().lower() for kv in f]
    try:
        chars.remove('')
    except ValueError:
        pass
    idx2char = dict(enumerate(sorted(boundary_signs.values()) + sorted(set(chars)), 1))
    print(idx2char)
    char2idx = reverse_dic(idx2char)
    char_types = np.string_(itemgetter(*sorted(idx2char.keys()))(idx2char))

    if glob(title_file):
        with open(title_file, 'r') as f:
            title_utts = np.array([re.sub(r'\t\w+$', '', kv) for kv in f]).astype(int)
        title_utts = np.insert(np.cumsum(title_utts), 0, 0)
        assert title_utts[-1] + 1 == len(utt_id)
        title_id = utt_id[title_utts]
    else:
        title_file = None

    sents_with_bnd = [BOS + BOW + (EOW + BOW).join(sent.split(' ')) + EOW + EOS for sent in all_sents]
    sent_id = get_begin_ids(sents_with_bnd)
    cat_upr_indictr = np.array([upper_or_lower(i) for sent in sents_with_bnd for i in sent], dtype=int)
    cat_encoded_seq = np.array([char2idx[i.lower()] for sent in sents_with_bnd for i in sent], dtype=int)
    shifted_bow_id = np.where(cat_encoded_seq == char2idx[BOW])[0]
    shifted_eow_id = np.where(cat_encoded_seq == char2idx[EOW])[0]

    # to examine if the same sentence is reconstructed from encoded array
    test_reconstruct = True
    if test_reconstruct:
        test_sid = np.random.randint(0, len(sent_id)-1, 3)
        for _i in test_sid:
            encoded_snt = cat_encoded_seq[sent_id[_i]:sent_id[_i+1]]
            is_upper = cat_upr_indictr[sent_id[_i]:sent_id[_i+1]]
            re_sent = arr2sent(encoded_snt, idx2char, i2bnd_dic=boundary_signs, up_indctr=is_upper)
            assert re_sent.rstrip() == all_sents[_i], \
                ('got reconstructed sentence: ', re_sent, '\nexpected:', all_sents[_i])

    with h5py.File(os.path.join(outdir, 'all_char.hdf5'), 'w') as f:
        f.create_dataset('char_types', data=char_types, dtype="S1")
        f.create_dataset('bndry_types', data=bndry_types, dtype='S3')
        # print(bndry_types.dtype)
        if title_file:
            f['title_id'] = title_id
        f['utt_id'] = utt_id
        f['bow_id'] = shifted_bow_id
        f['eow_id'] = shifted_eow_id
        f.create_dataset('sent_id', data=sent_id)
        f.create_dataset('cat_encoded_seq', data=cat_encoded_seq)
        f.create_dataset('cat_is_upper', data=cat_upr_indictr)
