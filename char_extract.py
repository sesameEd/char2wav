#!/usr/bin/python3
import argparse
import re
# from nltk import sent_tokenize, word_tokenize
import numpy as np
# import os
# import pandas as pd
# from glob import glob
# from scipy.io import wavfile

parser = argparse.ArgumentParser('Process and character array')
parser.add_argument('-t', '--alltxtpath', default='data/title/all.txt', type=str)
parser.add_argument('-d', '--dict', default='data/title/vocab.freq', type=str,
                    help='the tsv file storing all character types (with counts)')
# parser.add_argument("--wav_dir", default='data/wav', type=str, help='default to "data/normalized/"')
# parser.add_argument('-s', '--suffix', default='txt', type=str, help='suffix of script files')
# parser.add_argument('--frac', action='store_true', help='processes all file under current directory')
parser.add_argument('-n', '--titles', default='data/title/title.freq', type=str,
                    help='the tsv file storing the different titles (with counts)')
args = parser.parse_args()

def reverse_dic(input_dic):
    return {v:k for k,v in input_dic}

script_file = args['alltxtpath']
with open(script_file, 'r') as f:
    all_scrpts = [line.rstrip() for line in f]
utt_idx = np.array([len(line) - line.count(' ') for line in all_scrpts].insert(0, 0))
utt_idx = np.cumsum(utt_idx)

vocab_file = args['dict']
with open(vocab_file, 'r') as f:
    idx2char = [re.sub(r'^\d+\t', '', kv).rstrip().lower() for kv in f]
idx2char.remove(' ')
idx2char = dict(enumerate(sorted(set(idx2char))))
char2idx = reverse_dic(idx2char)

title_file = args['titles']
with open(title_file, 'r') as f:
    title_idx = np.array([re.sub(r'\t\w+$', '', kv) for kv in f]).astype(int)
title_idx = np.cumsum(title_idx)

# all_sents = sent_tokenize(' '.join(all_scrpts))
# bos_idx = np.array([len(sent) - sent.count(' ') for sent in all_sents].insert(0, 0))
# eos_idx = bos_idx - 1
# del bos_idx[-1]
# del eos_idx[0]

# run_tokenizer = ("java -cp $CORENLP_HOME/stanford-corenlp-3.9.1.jar"
#                  "edu.stanford.nlp.process.DocumentPreprocessor"
#                  "{0} > {0}.out".format(script_file))
# convert_quotes = ("sed -e s/\`\`/\\\"/g -e s/\\\'\\\'/\\\"/g -e s/\`/\\\'/g "
#                   "-i {}".format(script_file + '.out'))
# try:
#     subprocess.check_output(run_tokenizer)
# except CalledProcessError:
#     print("Run `echo $CORENLP_HOME` to check if environmental variable is set",
#           "to your local /some/path/to/stanford-corenlp-full-2018-02-27 directory; "
#           "if not, run `source setup.sh`")
# try:
#     subprocess.check_output(convert_quotes)
# except CalledProcessError:
#     pass

# all_words =
# wavdir = args['wav_dir']
# sfx = args['suffix']
# if args['frac']:
#     wavs = glob(os.path.join(wavdir, '*.wav'))
# else:
#     wavs = glob(os.path.join(wavdir, 'A*.wav'))
# wav_names = [os.path.basename(wav).split('.')[0] for wav in wavs]
