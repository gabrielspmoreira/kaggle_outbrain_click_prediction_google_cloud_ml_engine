#!/usr/bin/env bash
set -e

echo "Instaling extra Python modules..."

PIP=/opt/conda/bin/pip

$PIP install gcloud
$PIP install google_compute_engine
$PIP install numpy==1.11.2
$PIP install pandas==0.19.2
$PIP install scipy==0.18.1
$PIP install scikit-learn==0.18.1
$PIP install spark-sklearn
$PIP install matplotlib
$PIP install seaborn
$PIP install imbalanced-learn
$PIP install graphviz
$PIP install crcmod