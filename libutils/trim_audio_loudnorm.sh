#!/bin/bash

display_help() {
    echo "Usage: $0 [-h --help] [-o --outdir <output_wav_dir>] [-m --mode <mode>] [-r --sample_rate n] <input_wav_dir>" >&2
    echo
    echo "   -o, --outdir       output directory of the trimmed audios, defaults to <input_wav_dir>/../normalized"
    echo "   -m, --mode         operation to be conducted: either or both of the two:
                                trim: removes preceding silence of the audio files, helps align beginning of audio
                                normalize: performs loudness normalization using ffmpeg to bring the average amplitude of all audio files to the same level;
                                if both are selected, normalized wavs will overwrite the trimmed wavs in the outdir"
    echo "   -r, --sample_rate  The sampling rate of the wavs, defaults to 48000"
    # exit 0
}
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1; fi

OPTIONS=o:m:r:h
LONGOPTS=outdir:,mode:,sample_rate:,help
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

TRIM=true
NORMALIZE=true
SAMPLE_RATE=48000

while true; do
  case "$1" in
    -m|--mode)
      mode=$2
      case "$mode" in
        trim)
          NORMALIZE=false
          ;;
        normalize)
          TRIM=false
          ;;
        all)
          TRIM=true
          NORMALIZE=true
          ;;
        *)
          echo mode must be one of below: trim, normalize, or all
          ;;
        esac
      shift 2
      ;;
    -o|--outdir)
      outdir=$2
      shift 2
      ;;
    -r|--sample_rate)
      SAMPLE_RATE=$2
      shift 2
      ;;
    -h|--help)
      display_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Refer to --help for available options"
      exit 3
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
    echo "$0: A single input wavefile directory is required."
    exit 4; fi
wavdir=$1
if [ ! -v outdir ]; then
  outdir=${wavdir}/../normalized
  echo using default output directory: $outdir; fi

echo wav directory is: $wavdir
echo output files to: $outdir
echo doing: trimming? $TRIM. normalization? $NORMALIZE
echo using sample rate $SAMPLE_RATE

if [ ! -d "$outdir" ]; then
  mkdir $outdir
fi

if [ "$TRIM" = true ]; then
  for wav_file in `ls $wavdir/*`; do
    wn=`basename $wav_file .wav`
    sox $wav_file $outdir/${wn}.wav silence 1 0.1 0.06%
  done;
fi

if [ "$NORMALIZE" = true ]; then
  if [ "$TRIM" = true ]; then
    echo overwriting trimmed audio with normalized ones
    ffmpeg-normalize $wavdir/*.wav -ar $SAMPLE_RATE -f -of $outdir -ext wav
  else
    ffmpeg-normalize $wavdir/*.wav -ar $SAMPLE_RATE -f -of $outdir -ext wav; fi
fi
