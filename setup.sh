#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./magphase/src:.
if [ ! -d data ]; then
  mkdir data
fi
