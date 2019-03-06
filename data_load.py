#!/usr/bin/python3
import argparse
parser = argparse.ArgumentParser('Process and store data')
parser.add_argument("--wav_dir", default='../blzd/wav', type=str, help='default to "blzd" under parent directory')
parser.add_argument('--txt_dir', default='../blzd/txt', type=str)
parser.add_argument('--lab_4mt', default='txt', type=str, help='format of lab files')
args = parser.parse_args()
