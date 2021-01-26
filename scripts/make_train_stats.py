#!/usr/bin/env python3
"""
Usage: make_train_stats.py CSVFILE [options]

Arguments:
  CSVFILE          CSV file as produced by make_corpus_stats.py

Options:
  --conll2018 DIR       Directory for UD v2.2 treebanks (as used by
                        CoNLL 2018 Shared Task)
  --help                Help me!

"""

from collections import Counter
from docopt import docopt
import logging
import logzero
from logzero import logger as log
import os
from pathlib import Path


SEP = ","


def get_conll2018_train_stats(lang, udpath):
    train_sents, train_toks = 0, 0

    for trainfile in udpath.rglob(f"{lang}_*-ud-train.conllu"):
        with open(trainfile, "r") as f:
            for line in f:
                if line.strip():
                    train_toks += 1
                else:
                    train_sents += 1

    return train_sents, train_toks


if __name__ == "__main__":
    args = docopt(__doc__)

    data = []
    with open(args["CSVFILE"], "r") as f:
        columns = f.readline().strip().split(SEP)
        for line in f:
            if not line.strip():
                continue
            line = line.strip().split(SEP)
            stats = {
                col: line[i]
                for i, col in enumerate(columns)
            }
            data.append(stats)

    for col in ("train_sents", "train_toks"):
        if col not in columns:
            columns.append(col)

    if args["--conll2018"]:
        udpath = Path(args["--conll2018"])
        stats_by_lang = {}
        for stats in data:
            if stats["dataset"] != "conll2018":
                continue
            corpus = stats["file"].split(".")[0]
            lang = corpus.split("_")[0]

            if lang not in stats_by_lang:
                stats_by_lang[lang] = get_conll2018_train_stats(lang, udpath)
            train_sents, train_toks = stats_by_lang[lang]

            stats["train_sents"] = train_sents
            stats["train_toks"] = train_toks

    print(SEP.join(columns))
    for stats in data:
        print(SEP.join(str(stats.get(col, '')) for col in columns))
