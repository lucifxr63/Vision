#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/Train data/Val data/Test
mkdir -p models artifacts weights

echo "Entorno listo. Coloca tus datos en ./data y ejecuta:"
echo "python train_eval.py --train_dir ./data/Train --val_dir ./data/Val --test_dir ./data/Test"
