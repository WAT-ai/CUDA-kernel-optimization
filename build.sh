#!/bin/bash

# conda env setup
conda create -n llm_optim python=3.10
conda activate llm_optim

REPO_URL="https://github.com/WAT-ai/kernel_tuner"
git clone $REPO_URL

python -m pip install -U setuptools pip
pip install -r requirements.txt
