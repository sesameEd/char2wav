#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./magphase/src:.
export CORENLP_HOME=`realpath -s stanford-corenlp-full-2018-02-27`
for file in `find . -name "*.jar"`; do
  export CLASSPATH="$CLASSPATH:`realpath $file`"
done
if [ ! -d data ]; then
  mkdir data
fi
