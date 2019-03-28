#!/bin/bash
GET_MAGPHASE=true

TOKENIZE=true
LOUDNORM=true

git submodule update --init --recursive

if [ "$TOKENIZE" = true ]; then
  echo compiling tokenization tools: Stanford CoreNLP
  GET_CORENLP=true
  GET_NLTK=false
else
  GET_CORENLP=false
  GET_NLTK=false
fi

if [ "$LOUDNORM" = true ]; then
  echo compiling loudnorm tools: ffmpeg, ffmpeg-normalize
  GET_FFMPEG=true
  GET_FMN=true
else
  GET_FFMPEG=false
  GET_FMN=false
fi

if [ "$GET_MAGPHASE" = true ]; then
  echo ===============================compiling magphase=============================
  cd magphase/tools
  ./download_and_compile_tools.sh
  cd ../../
fi

if [ "$GET_NLTK" = true ]; then
  echo ===============================installing NLTK=============================
  pip install --user nltk
  python -m nltk.downloader punkt
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
  cd ffmpeg
  ./configure
  echo debug: if you received error message:
  echo \"nasm/yasm not found or too old. Use \-\-disable-x86asm for a crippled build.\"
  echo run \"sudo apt-get install yasm\"
  make
  make install
fi

if [ "$GET_FMN" = true ]; then
  echo ==========================compiling ffmpeg-normalize==========================
  cd ffmpeg-normalize
  pip install --user .
fi

source setup.sh
