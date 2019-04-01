#!/usr/bin/python3
import argparse
import re
import h5py
from operator import itemgetter
import numpy as np
import os

parser = argparse.ArgumentParser('Process transcripts and convert to index-encode array')
parser.add_argument('-a', '--alltxtpath', default='data/title/all.txt', type=str,
                    help=('the file that stores all transcripts with utterances separated by \\n. '
                          'Tokenized transcript is assumed to be in the same dir with "tknzd_" prefix. '
                          'Output .hdf5 file will be stored in its parent directory if not specified. '))
parser.add_argument('-d', '--dict', default='data/title/vocab.freq', type=str,
                    help='the tsv file storing all character types (with counts)')
parser.add_argument('-t', '--titles', default='data/title/title.freq', type=str,
                    help='the tsv file storing the different titles (with counts)')
parser.add_argument('-o', '--outdir', type=str, help='the directory to save output .h5 file')
args = vars(parser.parse_args())
do_align = True
boundary_signs = dict(zip(['BOS', 'BOW', 'EOS', 'EOW'], map(str, range(4))))
BOS, BOW, EOS, EOW = map(str, range(4))
boundary_arr = np.string_(['BOS', 'BOW', 'EOS', 'EOW'])

vocab_file = args['dict']
title_file = args['titles']
script_file = args['alltxtpath']
dir_name, file_name = os.path.split(script_file)
tknzd_file = os.path.join(dir_name, 'tknzd_' + file_name)
outdir = args.get('outdir')
if not outdir:
    outdir = os.path.dirname(dir_name)
# print(outdir)
# print(boundary_signs.get('outdir', os.path.dirname(dir_name)))
# quit(0)

def reverse_dic(input_dic):
    return {v:k for k,v in input_dic.items()}

def get_begin_ids(list_sents):
    """
    returns an array of beginning indices of sentences (with spaces removed);
    an extra index of the length of the entire concatenated sentence array is added;
    s.t. the individual sents can be indexed by concat_array[id[i]:id[i+1]]
    """
    lens = np.array([len(line) - line.count(' ') for line in list_sents])
    return np.cumsum(np.insert(lens, 0, 0),0)

def get_bow_eow(list_sents):
    lens = np.array([len(word) for sent in list_sents for word in sent.split(' ')])
    lens = np.cumsum(np.insert(lens, 0, 0), 0)
    bow = np.delete(lens, -1, 0)
    eow = np.delete(lens-1, 0, 0)
    return bow, eow

def upper_or_lower(char):
    if char.isalpha():
        return 1 if char.isupper() else 0
    else:
        return 2

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
        _ids = np.where(up_indctr==1)[0]
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
    utt_idx = get_begin_ids(all_scripts)

    if not os.path.isfile(tknzd_file):
        print("Cannot find tokenized scripts in {0}, will assume {1} to be tokenized.".format(tknzd_file, script_file))
        tknzd_file = script_file
        do_align = False
    with open(tknzd_file, 'r') as f:
        all_sents = [line.rstrip() for line in f]
    bos_id = get_begin_ids(all_sents)
    eos_id = np.delete(bos_id-1, 0, 0)
    bos_id = np.delete(bos_id, -1, 0)
    bow_id, eow_id = get_bow_eow(all_sents)
    assert eos_id[-1] == eow_id[-1], 'did not get the right sentence or word boundaries'
    assert eos_id[-1] + 1 == utt_idx[-1], 'index of last character does not accord'
    assert all(np.in1d(utt_idx[:-1], bow_id, assume_unique=True)), 'some utts didn\'t end with a whole word?'

    if do_align:
        ebows_inplace = np.sum([bin_annotate_1d(ids, utt_idx[-1])
                                for ids in [bos_id, eos_id, bow_id, eow_id]], 0)
        assert ebows_inplace.shape == (utt_idx[-1],), 'Error with annotating 1d-array'
        shifted_uid = shift_indices(utt_idx, ebows_inplace)

    with open(vocab_file, 'r') as f:
        chars = [re.sub(r'^\d+\t', '', kv).rstrip().lower() for kv in f]
    chars.remove('')
    idx2char = dict(enumerate(sorted(boundary_signs.values()) + sorted(set(chars))))
    char2idx = reverse_dic(idx2char)
    char_arr = np.string_(itemgetter(*sorted(idx2char.keys()))(idx2char))

    with open(title_file, 'r') as f:
        title_idx = np.array([re.sub(r'\t\w+$', '', kv) for kv in f]).astype(int)
    title_idx = np.insert(np.cumsum(title_idx), 0, 0)
    assert title_idx[-1] + 1 == len(shifted_uid)

    sents_with_bnd = [BOS + BOW + (EOW + BOW).join(sent.split(' ')) + EOW + EOS for sent in all_sents]
    sent_idx = get_begin_ids(sents_with_bnd)
    if not do_align:
        utt_idx = sent_idx
    sent_idx = get_begin_ids(sents_with_bnd)
    cat_upr_indictr = np.array([upper_or_lower(i) for sent in sents_with_bnd for i in sent], dtype=int)
    cat_encoded_seq = np.array([char2idx[i.lower()] for sent in sents_with_bnd for i in sent], dtype=int)
    assert cat_upr_indictr.shape == cat_encoded_seq.shape

    test_sid = np.random.randint(0, len(sent_idx)-1, 3)
    for _i in test_sid:
        encoded_snt = cat_encoded_seq[sent_idx[_i]:sent_idx[_i+1]]
        upper_indctr = cat_upr_indictr[sent_idx[_i]:sent_idx[_i+1]]
        re_sent = arr2sent(encoded_snt, idx2char, i2bnd_dic=boundary_signs, up_indctr=upper_indctr)
    #     print(all_sents[_i])
        assert re_sent[:-1] == all_sents[_i], print(re_sent, all_sents[_i])

    with h5py.File(os.path.join(outdir, 'all_char.h5'), 'w') as f:
        f.create_dataset('char_arr', data=char_arr)
        f.create_dataset('boundary_arr', data=boundary_arr)
        f.create_dataset('title_idx', data=title_idx)
        f.create_dataset('utt_idx', data=shifted_uid)
        f.create_dataset('sent_idx', data=sent_idx)
        f.create_dataset('cat_encoded_seq', data=cat_encoded_seq)
        f.create_dataset('cat_upr_indictr', data=cat_upr_indictr)
