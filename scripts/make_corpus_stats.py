#!/usr/bin/env python3
"""
Usage: make_corpus_stats.py DIRECTORY [options]

Arguments:
  DIRECTORY        Path to directory of processed files
                   (i.e. produced by process_*.py scripts)

Options:
  --help           Help me!

"""

from collections import Counter
from docopt import docopt
import gzip
import logging
import logzero
from logzero import logger as log
import os
from pathlib import Path
import pyconll
from tqdm import tqdm


SEP = ","


if __name__ == "__main__":
    args = docopt(__doc__)
    allfiles = list(Path(args["DIRECTORY"]).rglob('*.conllu.gz'))
    columns = ["dataset", "file", "ttr", "bor", "feats", "ftr"]

    print(SEP.join(columns))

    for filename in tqdm(allfiles):
        with gzip.open(filename, "rt") as f:
            conll = pyconll.load_from_string(f.read())

        p = Path(filename)
        stats = {
            "dataset": p.parent.name,
            "file": p.name,
        }
        tok_count = Counter()
        bad_count = Counter()
        feat_count = Counter()
        for sent in conll:
            for tok in sent:
                tok_count[tok.form.lower()] += 1
                bad_count[tok.misc["tag"].pop()] += 1
                feat_count[f"UPOS={tok.upos}"] += 1
                for feat, values in tok.feats.items():
                    feat_count[f"{feat}={','.join(sorted(values))}"] += 1

        # Type-token ratio
        stats["ttr"] = len(tok_count) / sum(tok_count.values())
        # Task performance (BAD/OK ratio)
        stats["bor"] = bad_count["BAD"] / sum(bad_count.values())
        # Feature count
        stats["feats"] = len(feat_count)
        # Feature ratio per million
        stats["ftr"] = 1000000 * len(feat_count) / sum(feat_count.values())

        print(SEP.join(str(stats[col]) for col in columns))
