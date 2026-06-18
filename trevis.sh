#!/bin/bash

set -e  

python scripts/random_trees.py
python scripts/tokenize.py
python scripts/ttvae_train.py
python scripts/gradient.py