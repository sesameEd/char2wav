#!/bin/bash
git submodule update --init --recursive
cd magphase/tools
./download_and_compile_tools.sh

cd ../../
source setup.sh
