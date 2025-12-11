#!/usr/bin/env bash
set -e

git init
git add README.md requirements.txt setup_git.sh train.py src
git commit -m "Initial full-quantum RL for 5-atom generation"
