# char2wav
- To download and compile all tools run
```shell
./compile_tools.sh -a
```
you also have the option of installing only the dependencies you need, 
- Set up environment:
```shell
source setup.sh
```
always run this before you do any experiment.
### Preparing data
*This part may be skipped if you wish to prepare your own data or have already extracted them in the format of hdf5.*
- to trim the audio of its initial silence
- to perform loudness normalization on all wav files (so the overall average perceived loudness of all audios are at the same level and the variation between from file to file is minimized), first make sure the following two dependencies: [`ffmpeg-normalize`](https://github.com/slhck/ffmpeg-normalize.git) and [`ffmpeg`](http://www.ffmpeg.org/) are successfully built and compiled with the `compile_tools.sh` scripts.
to perform two-pass loudness normalization on the wavfiles from a directory `<input_wav_dir>`, run:
```shell
ffmpeg-normalize input/wav/dir/*.wav -ar $SAMPLE_RATE -f -of output/wav/dir -ext wav
```
