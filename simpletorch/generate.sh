#!/bin/bash

if [ -e build ]
then
  rm -rf build
fi

mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=/Users/kevinlynna/github/pytorch/torch/lib/tmp_install ..

if [ $? -eq 0 ]
then
  make
fi
