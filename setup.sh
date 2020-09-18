#!/bin/bash
mkdir weights
mkdir weights/yolact
mkdir weights/DYnamicCNN
mkdir external

echo "Setting up envirnonment"
conda create -n DYnamic_CNN python=3.7 pip

echo "Installing Yolact"
cd external
git clone https://github.com/dbolya/yolact.git
cd ..
conda env update -n DYnamic_CNN -f env.yaml

echo "Compiling c++ code"
cmake src/cpp/CMakeLists.txt && make --directory=src/cpp
make src/cpp/


