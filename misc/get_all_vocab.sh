#!/bin/bash
######################################################################################
# This script processes the script files in various ways
# # # #   -m, --mode
# # # #       mode of operation:
# # # #       'getvocab' obtains a frequency list of characters in the entire script;
# # # #       'tokenize' tokenizes the scripts and pads with <bos> (beginning of sentence)
# # # #        and <eos>. Requires stanford coreNLP package pre-installed
######################################################################################
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=i:o:t
LONGOPTS=indir:,outdir:,bytitle
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

transdir="data/txt"
outdir="data/title"
# GETVOCAB=true
BYTITLE=false
SUFFIX=_[0-9]{3}_[0-9]{3}.txt
# TKNZ=false
display_help() {
    echo "Usage: $0 [-h --help] [-o --output <output_wav_dir>] [-m --mode <mode>] [-r --sample_rate n] <input_wav_dir>" >&2
    echo
    echo "     -i, --indir       location of the transcript files, defaults to 'data/txt'"
    echo "     -o, --outdir      output directory of the processed files (vocab list or tokenized texts), defaults to 'data/title'"
    echo "     -t, --bytitle     if switched on, will count the number of utterances for under title/chapters (if the files can be"
    echo "                       expressed as '<book/chapter/name><utt/idx>') and output to file 'data/title/title_count.txt'. e.g."
    echo "                       'AMidsummerNightsDream' <- 'AMidsummerNightsDream_000_000.txt'"
    echo "                       You have to replace the \$SUFFIX with a regex that matches"
    echo "                       your own naming standard and works for linux sed command"
}

while true; do
    case "$1" in
      # -m|--mode)
      #   mode=$2
      #   case "$mode" in
      #     getvocab)
      #       GETVOCAB=true
      #       ;;
      #     # tokenize)
      #     #   TKNZ=true
      #     #   ;;
      #     *)
      #       echo mode must be one of below: getvocab, tokenize
      #       ;;
      #   esac
      #   shift 2
      #   ;;
      -i|--indir)
        transdir=$2
        shift 2
        ;;
      -o|--outdir)
        outdir=$2
        shift 2
        ;;
      -t|--bytitle)
        BYTITLE=true
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
        echo "Programming error"
        exit 3
        ;;
    esac
done

# echo tokenization mode: $TKNZ
# echo get vocab mode: $GETVOCAB
echo input dir: $transdir
echo output dir: $outdir
# exit 0

if [ ! -d "$outdir" ]; then
  mkdir $outdir
fi

if [ "$BYTITLE" = true ]; then
  echo "===================:creating character vocab and title count:===================="
  echo -n "" >| ${outdir}/all.txt
  find ${transdir} -type f -printf "%f\n" | sed -r "s/${SUFFIX}//" | uniq -c >| ${outdir}/title_count.txt
  sed -e 's/^[ \t]*//' -e 's/ /\t/' -i ${outdir}/title_count.txt
  for book in `find ${transdir} -type f -printf "%f\n" | sed -r "s/${SUFFIX}//" | uniq`; do
    # echo $book
    echo -n "" >| ${outdir}/all_${book}.txt
    for f in `ls ${transdir}/${book}* | sort`; do
      echo `cat $f` >> ${outdir}/all_${book}.txt
      echo `cat $f` >> ${outdir}/all.txt
    done
  done
  od -cvAn -w1 ${outdir}/all.txt | sort | uniq -c >| $outdir/all.vcb
  sed -e 's/^[ \t]*//' -re 's/[[:space:]]{2,4}/\t/g' -e '/[0-9]$/d' -e '/\\n$/d' -i ${outdir}/all.vcb
else
  echo "===================:creating character vocab list for text:===================="
  echo -n "" >| ${outdir}/all.txt
  for f in `ls ${transdir}/* | sort`; do
    echo `cat $f` >> ${outdir}/all.txt
  done
  od -cvAn -w1 ${outdir}/all.txt | sort | uniq -c >| $outdir/all.vcb
  sed -e 's/^[ \t]*//' -re 's/[[:space:]]{2,4}/\t/g' -e '/[0-9]$/d' -e '/\\n$/d' -i ${outdir}/all.vcb
fi

# if [ "$TKNZ" = true ]; then
#   echo "tokenizing lab files:========================================================"
#   source ~/.bashrc
#   java -cp $CORENLP_HOME/stanford-corenlp-3.9.1.jar \
#     edu.stanford.nlp.process.DocumentPreprocessor \
#     BHoT/bhot_text.txt > BHoT/bhot_tknz.txt
# fi
