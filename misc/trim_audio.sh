#!/bin/bash
######################################################################################
# Trims audios by removing starting silence from wav. Ending silences are kept as they
# are considered meaningful
#   -i, --indir
#       location of the input wav files, defaults to '../blzd/wav'
#   -o, --outdir
#       output directory of the trimmed audios defaults to '../blzd/trimmed'
######################################################################################
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1; fi

OPTIONS=i:o:
LONGOPTS=outdir:,outdir:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

wavdir="../blzd/wav"
outdir="../blzd/trimmed"

while true; do
  case "$1" in
    -i|--indir)
      wavdir=$2
      shift 2
      ;;
    -o|--outdir)
      outdir=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Programming error"
      exit 3
      ;;
  esac
done

echo wav directory is: $wavdir
echo output trimmed files to: $outdir

if [ ! -d "$outdir" ]; then
  mkdir $outdir
fi

for wav_file in `ls $wavdir/*`; do
  wn=`basename $wav_file .wav`
  sox $wav_file $outdir/${wn}.wav silence 1 0.1 0.06%
done
