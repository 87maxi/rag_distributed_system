#!/bin/bash


rm -rf ~/vllm-local

uv init ~/vllm-local
cd ~/vllm-local && uv venv 

source .venv/bin/activate

uv pip install  vllm 

uv pip install diffusers accelerate transformers
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
