#!/bin/bash
git submodule update --init --recursive

echo ===============================compiling magphase=============================
cd magphase/tools
./download_and_compile_tools.sh
cd ../../

echo ================================compiling ffmpeg==============================
cd ffmpeg
./configure
echo debug: if you received error message:
echo \"nasm/yasm not found or too old. Use \-\-disable-x86asm for a crippled build.\"
echo run \"sudo apt-get install yasm\"
make
make install

source setup.sh
