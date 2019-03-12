#!/bin/bash
######################################################################################
# This script processes the script files in various ways
#   -m, --mode
#       mode of operation:
#       'getvocab' obtains a frequency list of characters in the entire script;
#       'tokenize' tokenizes the scripts and pads with <bos> (beginning of sentence)
#        and <eos>. Requires stanford coreNLP package pre-installed
#   -i, --indir
#       location of the script files, defaults to '../blzd/txt'
#   -o, --outdir
#       output directory of the processed files (vocab list or tokenized texts), def-
#       aults to './data'
######################################################################################
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=i:m:o:
LONGOPTS=indir:,mode:,outdir:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

labdir="../blzd/txt"
outdir="./data"
GETVOCAB=false
TKNZ=false

while true; do
    case "$1" in
      -m|--mode)
        mode=$2
        case "$mode" in
          getvocab)
            GETVOCAB=true
            ;;
          tokenize)
            TKNZ=true
            ;;
          *)
            echo mode must be one of below: getvocab, tokenize
            ;;
        esac
        shift 2
        ;;
      -i|--indir)
        labdir=$2
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

echo tokenization mode: $TKNZ
echo get vocab mode: $GETVOCAB
echo input dir: $labdir
echo output dir: $outdir

if [ ! -d "$outdir" ]; then
  mkdir $outdir
fi

if [ "$GETVOCAB" = true ]; then
  echo "creating character vocab list for text:======================================="
  echo -n "" >| ${outdir}/all.txt
  if [ ! -e ${outdir}/title_count.txt ]; then
    ls ${labdir}/ | sed 's/_[0-9]+_[0-9]+.txt//' | uniq -c > ${outdir}/title_count.txt
    sed -e 's/^[ \t]*//' -e 's/ /\t/' -i ${outdir}/title_count.txt; fi
  # for name in `ls ${labdir}/*_000_000.txt`; do
  for book in `ls ${labdir} | sed 's/_[0-9]+_[0-9]+.txt//' | uniq`
    # book=`basename ${name} _000_000.txt`
    echo $book
    echo -n "" > ${outdir}/all_${book}.txt
    for f in `ls ${labdir}/${book}_*.txt | sort`; do
      echo `cat $f` >> ${outdir}/all_${book}.txt
      echo `cat $f` >> ${outdir}/all.txt
      #
      # echo `cat $f` | tr '[:upper:]' '[:lower:]' >> ${outdir}/all_${book}.txt
      # echo `cat $f` | tr '[:upper:]' '[:lower:]' >> ${outdir}/all.txt
    done
  done
  od -cvAnone -w1 ${outdir}/all.txt | sort | uniq -c > $outdir/all.vcb
fi

if [ "$TKNZ" = true ]; then
  echo "tokenizing lab files:========================================================"
  source ~/.bashrc
  java -cp $CORENLP_HOME/stanford-corenlp-3.9.1.jar \
    edu.stanford.nlp.process.DocumentPreprocessor \
    BHoT/bhot_text.txt > BHoT/bhot_tknz.txt
fi
