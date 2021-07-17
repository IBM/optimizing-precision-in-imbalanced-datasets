#!/bin/bash
echo '[ImbOpt Sample Dataset Setup]'

# Generate Data Folder (If Not Exists)
echo '> Generating Data Folder'
mkdir -p data/ && cd data/
echo ''

# Download Datasets
echo '[Downloading]: Scene Dataset'
mkdir -p scene/ && cd scene/
wget https://www.openml.org/data/get_csv/1390080/phpuZu33P
mv phpuZu33P scene.csv
cd ..
echo ''

echo '[Downloading]: Protein Dataset'
mkdir -p protein/ && cd protein/
wget https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz
tar -xzf data_kddcup04.tar.gz
rm data_kddcup04.tar.gz
cd ..
echo ''

echo '[Downloading]: Optical Digits Dataset'
mkdir -p optical_digit/ && cd optical_digit/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra
wget https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes
cd ..
echo ''

echo '[Downloading]: Thydroid Dataset'
mkdir -p thyroid/ && cd thyroid/
mkdir -p bp/ && cd bp/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.test
cd ..
mkdir -p hyper/ && cd hyper/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.test
cd ..
mkdir -p hypo/ && cd hypo/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.test
cd ..
mkdir -p rep/ && cd rep/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.test
cd ..
mkdir -p dis/ && cd dis/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.test
cd ..
mkdir -p sick/ && cd sick/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.test
cd ../..
echo ''

echo '[Downloading]: Satelite Imagery Dataset'
mkdir satelite/ && cd satelite/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst
cd ..
echo ''

echo '[Downloading]: Spectrometer Dataset'
mkdir spectrometer/ && cd spectrometer/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/spectrometer/lrs.data
cd ..
echo ''

echo '[Downloading]: Wine Dataset'
mkdir wine/ && cd wine/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
cd ..
echo ''
