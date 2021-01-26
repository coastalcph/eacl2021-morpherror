#!/usr/bin/env python3
"""
Usage: add_feature_freqs.py CONLLFILE CSVFILE [options]

Add feature frequency information to analyzed CSV files.

Options:
  -n, --control-feats           Also generate non-morphological features as a control.
  -M, --no-morph-feats          Don't generate ANY morphological features.
  -I, --no-immediate-context    Don't generate features for preceding/following tokens.
  -L, --no-local-combinations   Don't generate feature combinations for the same token.
  -w, --wide-context NUM        Number of context tokens to include for wide context
                                features. [default: 4]

"""

import csv
from docopt import docopt
import gzip
import logging
import logzero
from logzero import logger as log
import pyconll
import os
import sys
from tqdm import tqdm
import warnings

import numpy as np
from utils.features import extract_features

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
logzero.loglevel(logging.DEBUG)
np.random.seed(2300)


if __name__ == "__main__":
    args = docopt(__doc__)

    log.info(f"Loading file: {args['CONLLFILE']}")
    if args["CONLLFILE"].endswith(".gz"):
        with gzip.open(args["CONLLFILE"], "rt") as f:
            conll = pyconll.load_from_string(f.read())
    else:
        conll = pyconll.load_from_file(args["CONLLFILE"])

    log.info(f"Loading CSV file: {args['CSVFILE']}")
    with open(args['CSVFILE'], "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        all_features = [line["feature_name"] for line in r]

    log.info(f"Extracting and counting features...")
    X_f, y_l = extract_features(
        conll,
        immediate_context=not bool(args["--no-immediate-context"]),
        wide_context=int(args["--wide-context"]),
        local_combinations=not bool(args["--no-local-combinations"]),
        control_features=bool(args["--control-feats"]),
        morph_features=not bool(args["--no-morph-feats"]),
    )

    freqs = [
        sum(feature in x for x in X_f)
        for feature in all_features
    ]

    if not all(freqs):
        log.warning("Some features were not found -- maybe wrong extraction settings?")

    with open(args['CSVFILE'], "r") as f:
        lines = f.readlines()

    assert len(lines) == len(freqs) + 1

    with open(args['CSVFILE'], "w") as f:
        header = lines.pop(0)
        print("\t".join((header.rstrip('\n'), "feat_count")), file=f)
        for line, count in zip(lines, freqs):
            print("\t".join((line.rstrip('\n'), f"{count}")), file=f)
