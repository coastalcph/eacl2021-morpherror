#!/bin/bash

shopt -s nullglob
shopt -s globstar

ANALYZE=./analyze.py
ADDFREQS=./add_feature_freqs.py
OUTDIR=../data/analyzed_rf
OUTDIR_NM=../data/analyzed_rf_nomorph

for f in ../processed/**/*.gz ; do
    fdir=$(basename $(dirname "$f"))
    fname=$(basename "$f")
    mkdir -p "$OUTDIR"/"$fdir"
    outname="$OUTDIR"/"$fdir"/"${fname%.conllu.gz}".tsv
    python $ANALYZE "$f" -n -I -L -w 0 --method drop-category-upos --log "$OUTDIR"/analyzed.log > "$outname"
    python $ADDFREQS -n -I -L -w 0 "$f" "$outname"
    mkdir -p "$OUTDIR_NM"/"$fdir"
    outname="$OUTDIR_NM"/"$fdir"/"${fname%.conllu.gz}".tsv
    python $ANALYZE "$f" -n -I -L -w 0 -M --method drop-category --log "$OUTDIR"/analyzed_nomorph.log > "$outname"
    python $ADDFREQS -n -I -L -w 0 -M "$f" "$outname"
done
