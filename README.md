# char2wav
This repo is a realization of an end-to-end text-to-speech system based largely on `char2wav` from ([Sotelo et al, 2017](https://openreview.net/pdf?id=B1VWyySKx), including model scripts, training scripts, and other scripts for building dependencies and for processing the data. \
The model is a bi-directional attention-based recurrent sequence generator (bi-ARSG) that takes texts of utterances (in the form of character strings) as input and predict a sequence of vocoder outputs, which are then translated into waveform. The vocoder features used here are [melgan-neurips](https://github.com/descriptinc/melgan-neurips/tree/6488045bfba1975602288de07a58570c7b4d66ea). The neural network is written with [PyTorch](https://pytorch.org/) framework. \

## Dependency
To download and compile all tools (except for PyTorch) run
```shell
./compile_tools.sh -a
```
if you want to install only some parts of the dependencies, run `./compile_tools.sh -h`
to get options. It'll also be reminded when using some function from that package.   
- To download and install PyTorch (must!), visit [pytorch instruction](https://pytorch.org/get-started/locally/) for the version that's suitable for you (options include: python version, OS, CUDA version, etc.).
```shell
pip install future tb-nightly soundfile numpy h5py tqdm
```
- To set up environment:
```shell
source setup.sh
```
*always* run this before any experiment.
### Preparing data
<!-- *This part may be skipped if you wish to prepare your own data or have already extracted them in the format of hdf5.* -->
The two shell scripts under directory `./libutils` is responsible for pre-processing the data (exactly how will be specified below).
The python scripts `voc_extract.py` and `char_extract.py` are for extracting the numeric
representations of the data (magphase vocoder features for audio files, and
one-hot encoded character sequence for transcripts).

Data management: the scripts assume the input files to be under directory
`data/<type/of/file>` and also output files accordingly:
ground-truth vocoder features under `data/vocoder`, loud-normed
wav files under `data/normalized`. So for convenience, copy the wavs and transcripts to
corresponding directories:
```shell
mkdir data/wav data/txt
cp -r <path/to/audio/files> data/wav
cp -r <path/to/transcripts> data/txt
```
You can always specify your own paths with the flags, `-o` or `--outdir` for output directory
and `-i` or `--indir` for input file directories (except for the `voc_extract.py` script);
or refer to the helper documentation with `--help` flag.

#### Pre-processing audio files
- to install the dependencies needed for loudnorm, run
```shell
./compile_tools.sh --loudnorm
```
- to trim initial silence and to perform loudness normalization on audio files
```shell
./libutils/trim_audio_loudnorm.sh data/wav
```
The presumed sampling rate of the audios is 48000, if that's not the case for your
files, specify the sampling rate with flag `-r` or `--sample_rate`:
```shell
./libutils/trim_audio_loudnorm.sh data/wav -r 48000
```
<!-- - to perform loudness normalization on wav files (so the overall average perceived loudness of all audios are at the same level and the variation between from file to file is minimized), first make sure the following two dependencies: [`ffmpeg-normalize`](https://github.com/slhck/ffmpeg-normalize.git) and [`ffmpeg`](http://www.ffmpeg.org/) are successfully built and compiled with the `compile_tools.sh` scripts.
to perform two-pass loudness normalization on the wavfiles from a directory `<input_wav_dir>`, run:
```shell
ffmpeg-normalize input/wav/dir/*.wav -ar $SAMPLE_RATE -f -of output/wav/dir -ext wav
``` -->
#### Pre-processing transcripts
- install tokenization package with
```shell
./compile_tools.sh --tokenize
```
- to collect transcripts into one file and tokenize it
```shell
./libutils/collect_vocab_tokenize.sh
```
#### Extracting ground truth vocoder features
- install vocoder dependency with
```shell
./compile_tools.sh -v
```
- to extract vocoder features. Specify the sampling rate with `-r` or `--sample_rate`
if it's not 48000
```shell
./voc_extract.py
```
#### Extracting character sequences
- ./char_extract.py
#### Train and SRNN:
```shell
./train_srnn.py -e 10 -R 2 2 4 -d 0.5 -batch_size 32 --learning_rate 4e-5
```
