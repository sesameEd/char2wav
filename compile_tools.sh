#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi
OPTIONS=tlva
LONGOPTS=tokenize,loudnorm,vocoder,all
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

display_help() {
    echo "Usage: $0 [-t --tokenize] [-l --loudnorm] [-v --vocoder] [-a --all]" >&2
    echo "-t, --tokenize       compiles tokenization tools: Stanford CoreNLP"
    echo "-l, --loudnorm       compiles loudnorm tools: ffmpeg, ffmpeg-normalize"
    echo "-v, --vocoder        compiles vocoder scheme: magphase"
    echo "-a, --all            compiles all of the above dependencies"
}


GET_CORENLP=false
GET_FFMPEG=false
GET_FMN=false
GET_MAGPHASE=false

while true; do
    case "$1" in
      -t|--tokenize)
        echo compiling tokenization tools: Stanford CoreNLP
        GET_CORENLP=true
        shift
        ;;
      -l|--loudnorm)
        echo compiling loudnorm tools: ffmpeg, ffmpeg-normalize
        GET_FFMPEG=true
        GET_FMN=true
        shift
        ;;
      -v|--vocoder)
        echo compiling vocoder scheme: magphase
        GET_MAGPHASE=true
        shift
        ;;
      -a|--all)
        echo compiling all dependencies: magphase, stanford CoreNLP, ffmpeg, and ffmpeg-normalize
        shift
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
        echo "Wrong flag, please refer to helper"
        exit 3
        ;;
    esac
done

pip install --user -r requirements.txt

if [ "$GET_MAGPHASE" = true ]; then
  echo ===============================compiling magphase=============================
  git submodule update --init magphase
  cd magphase/tools
  ./download_and_compile_tools.sh
  cd ../../
fi

if [ "$GET_CORENLP" = true ]; then
  echo ===========================compiling Stanford CoreNLP=========================
  wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
  unzip stanford-corenlp-full-2018-02-27.zip
  cd stanford-corenlp-full-2018-02-27
  for file in `find . -name "*.jar"`; do
    export CLASSPATH="$CLASSPATH:`realpath $file`"
  done
  cd ../
  export CORENLP_HOME=`realpath -s stanford-corenlp-full-2018-02-27`
fi

if [ "$GET_FFMPEG" = true ]; then
  echo ================================compiling ffmpeg==============================
  git submodule update --init ffmpeg
  cd ffmpeg
  ./configure --disable-ffplay --disable-ffprobe --disable-avdevice --disable-avcodec \
    --disable-avformat --disable-swresample --disable-swscale --disable-postproc
  if [ $? -eq 0 ]; then
    echo if you received error message concerning \"yasm\", run \"sudo apt-get install yasm\" and retry
  fi
  make
  make install
  if [ $? -eq 0 ]; then
    echo ffmpeg successfully installed, good to go!
  fi
fi

if [ "$GET_FMN" = true ]; then
  echo ==========================compiling ffmpeg-normalize==========================
  git submodule update --init ffmpeg-normalize
  cd ffmpeg-normalize
  pip install --user .
fi

# source setup.sh
