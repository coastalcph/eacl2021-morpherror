#!/bin/bash

set -o errexit

# from <http://www.statmt.org/wmt19/qe-task.html>
mkdir -p wmt19-qe
cd wmt19-qe
for f in task1_en-de_traindev.tar.gz task1_en-ru_traindev.tar.gz; do
    mkdir ${f%_*}
    wget https://deep-spin.github.io/docs/data/wmt2019_qe/$f
    tar xzf $f -C ${f%_*}/
    rm -f $f
done
