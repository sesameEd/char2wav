#!/usr/bin/python3
import h5py
import torch
from model import init_weights, shape_assert, Char2Voc
import argparse

parser = argparse.ArgumentParser('train Char2Voc model and pre-train character level embeddings')

if __name__ == '__main__':
    with h5py.File('')
