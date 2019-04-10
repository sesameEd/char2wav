#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=i:o:tm:
LONGOPTS=indir:,outdir:,bytitle,mode:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2; fi
eval set -- "$PARSED"

transdir="data/txt"
outdir="data/title"
GETVOCAB=true
TKNZ=true
BYTITLE=false
SUFFIX=_[0-9]{3}_[0-9]{3}.txt
display_help() {
    echo "Usage: $0 [-h --help] [-i --indir <input/transcript>] [-o --output <output_wav_dir>]
          [-m --mode getvocab|tokenize|all] [-t --bytitle]" >&2
    echo
    echo "     -i, --indir       location of the transcript files, defaults to 'data/txt'"
    echo "     -o, --outdir      output directory of the processed files (vocab list or tokenized texts), defaults to 'data/title'"
    echo "     -t, --bytitle     if switched on, will count the number of utterances for under title/chapters (if the files follows"
    echo "                       the form '<book_or_chapter_name><utt_id>') and output to file 'data/title/title.freq'. e.g."
    echo "                       'AMidsummerNightsDream' <- 'AMidsummerNightsDream_000_000.txt'"
    echo "                       You have to replace the \$SUFFIX with a regex that matches"
    echo "                       your own naming standard and works for linux sed command"
}

while true; do
    case "$1" in
      -m|--mode)
        mode=$2
        case "$mode" in
          getvocab)
            TKNZ=false
            ;;
          tokenize)
            GETVOCAB=false
            ;;
          all)
            TKNZ=true
            GETVOCAB=true
            ;;
          *)
            echo mode must be one of below: getvocab, tokenize
            ;;
        esac
        shift 2
        ;;
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

echo input dir: $transdir
echo output dir: $outdir
# exit 0

if [ ! -d "$outdir" ]; then
  mkdir $outdir
fi

if [ "$GETVOCAB" = true ]; then
  if [ "$BYTITLE" = true ]; then
    echo "===================:creating character vocab and title count:===================="
    echo -n "" >| ${outdir}/all.txt
    find ${transdir} -type f -printf "%f\n" | sort | sed -r "s/${SUFFIX}//" | uniq -c >| ${outdir}/title.freq
    sed -e 's/^[ \t]*//' -e 's/ /\t/' -i ${outdir}/title.freq
    if [ $? -eq 0 ]; then
      echo "Succesfully counted utterances under each title and stored in tab-separated file ${outdir}/title.freq"
    fi
    for book in `find ${transdir} -type f -printf "%f\n" | sort | sed -r "s/${SUFFIX}//" | uniq`; do
      # echo $book
      echo -n "" >| ${outdir}/all_${book}.txt
      for f in `ls ${transdir}/${book}* | sort`; do
        echo `cat $f` >> ${outdir}/all_${book}.txt
        echo `cat $f` >> ${outdir}/all.txt
      done
    done
    sed -z 's/\([a-z]\)\n\([A-Z]\)/\1.\n\2/g' -i ${outdir}/all.txt
    awk '{print length}' ${outdir}/all.txt | sort -g | uniq -c | tail -10
    if [ $? -eq 0 ]; then
      echo "All transcripts are collected in order and stored in file ${outdir}/all.txt"
    fi
    od -cvAn -w1 ${outdir}/all.txt | sort | uniq -c >| $outdir/vocab.freq
    sed -e 's/^[ \t]*//' -re 's/[[:space:]]{2,4}/\t/g' -e '/[0-9]$/d' -e '/\\n$/d' -i ${outdir}/vocab.freq
    if [ $? -eq 0 ]; then
      echo "A tab-separated table of character types and their occurrences is stored in ${outdir}/vocab.freq"
    fi

  else
    echo "===================:creating character vocab list for text:===================="
    echo -n "" >| ${outdir}/all.txt
    for f in `ls ${transdir}/* | sort`; do
      echo `cat $f` >> ${outdir}/all.txt
    done
    # sed -z 's/\([a-z]\)\n\([A-Z]\)/\1.\n\2/g' -i ${outdir}/all.txt
    # adds a period to lines without punctuation but with next line first char capitalized
    # awk '{print length}' ${outdir}/all.txt | sort -g | uniq -c | tail -10 #to get
    # gets the lengths of longest lines and their frequencies
    echo "getting a table of character types and their occurrences"
    if [ $? -eq 0 ]; then
      echo "A tab-separated table of character types and their occurrences is stored in ${outdir}/vocab.freq"
    fi
    od -cvAn -w1 ${outdir}/all.txt | sort | uniq -c >| $outdir/vocab.freq
    sed -e 's/^[ \t]*//' -re 's/[[:space:]]{2,4}/\t/g' -e '/[0-9]$/d' -e '/\\n$/d' -i ${outdir}/vocab.freq
  fi
fi

if [ "$TKNZ" = true ]; then
  echo "=============================:tokenizing transcripts:==========================="
  java -cp $CORENLP_HOME/stanford-corenlp-3.9.1.jar edu.stanford.nlp.process.DocumentPreprocessor \
    ${outdir}/all.txt >| ${outdir}/tknzd_all.txt
  echo ""
  sed -e s/\`\`/\"/g -e s/\'\'/\"/g -e s/\`/\'/g -e 's/-LRB-/\(/g' -e 's/-RRB-/\)/g' \
    -ze 's/\.\.\.\ "\ /...\ "\n/g' -ze 's/\.\.\.\ \([^"]\)/...\n\1/g' -i ${outdir}/tknzd_all.txt
  od -cvAn -w1 ${outdir}/tknzd_all.txt | sort | uniq -c >| $outdir/tknzd.freq
  sed -e 's/^[ \t]*//' -re 's/[[:space:]]{2,4}/\t/g' -e '/[0-9]$/d' -e '/\\n$/d' -i ${outdir}/tknzd.freq
  # awk '{print length}' ${outdir}/tknzd_all.txt | sort -g | uniq -c | tail -10
fi
