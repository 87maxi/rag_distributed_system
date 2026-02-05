#!/bin/bash
cd /tmp
rm -rf llama.cpp
cd /tmp &&    git clone https://github.com/ggerganov/llama.cpp
cd  /tmp/llama.cpp

# Crear directorio de build
rm -rf build/ &&  mkdir build && cd build

# Configurar con soporte CUDA para múltiples GPUs
CC=/bin/gcc-14 CXX=/bin/g++-14 cmake .. -DLLAMA_CUDA=ON  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DLLAMA_CUDA=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CUBLAS=OFF


# Compilar
cmake --build . --config Release -j$(nproc)

cd .. && mkdir -p  ~/bin/llama.cpp  && mv build ~/bin/llama.cpp

export PATH=$PATH:~/bin/llama.cpp/build/bin 

sudo cp  ~/bin/llama.cpp/build/bin/*.so*  /usr/local/lib



sudo ldconfig



#sudo ln -sf  ~/bin/llama.cpp/build/bin/llama-server /usr/local/bin/com.docker.llama-server


ln -sf ~/bin/llama.cpp/build/bin/llama-server ~/bin/llama.cpp/build/bin/com.docker.llama-server
