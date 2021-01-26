#!/usr/bin/env python3
"""
Usage: add_frequencies.py [options]

Add word frequency information to CoNLL files in MISC column.

Options:
  -f, --freqfile FREQFILE    JSON file with word frequencies by language.
                             [default: ../data/ud2.5-frequencies.json.gz]
  --debug                    Show debug-level output.
"""

from docopt import docopt
from glob import glob
import gzip
import json
import logging
import logzero
from logzero import logger as log
import numpy as np
import os
import pyconll
from pycountry import languages
from tqdm import tqdm
import unicodedata


SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = f"{SCRIPTDIR}/../data/processed/"


class FrequencyDict:
    def __init__(self, freqfile):
        with gzip.open(freqfile, "rt") as f:
            self.data = json.load(f)

    def get(self, lang, word):
        if lang not in self.data:
            raise ValueError(f"Unsupported language: {lang}")

        if not word:
            return 0

        word = unicodedata.normalize("NFC", word).lower()
        return self.data[lang].get(word, 0)

    def values(self, lang):
        return np.asarray(list(self.data[lang].values()))

    def __contains__(self, lang):
        return lang in self.data


def parse_language(name):
    lang = name.split(".")[0]
    if lang.startswith("task1"):
        lang = lang.split("-")[-1]
    elif "_" in lang:
        lang = lang.split("_")[0]
    return languages.lookup(lang)


def make_bin(bins, value):
    for (name, cutoff) in bins:
        if value >= cutoff:
            return name
    return None


def process_submissions(fd):
    for filename in tqdm(list(glob(f"{DATADIR}/**/*.conll*"))):
        language = parse_language(os.path.basename(filename))
        if getattr(language, "alpha_2", None) in fd:
            language = language.alpha_2
        elif getattr(language, "alpha_3", None) in fd:
            language = language.alpha_3
        else:
            log.error(f"Language '{language.name}' not in frequency dict!")
            continue

        freq_values = fd.values(language)
        freq_bins = (
            ("p99", round(np.percentile(freq_values, 99))),
            ("p98", round(np.percentile(freq_values, 98))),
            ("p95", round(np.percentile(freq_values, 95))),
            ("p90", round(np.percentile(freq_values, 90))),
            ("gt3", 4),
            ("rare", 0),
        )
        del freq_values

        if filename.endswith(".gz"):
            with gzip.open(filename, "rt") as f:
                conll = pyconll.load_from_string(f.read())
        else:
            conll = pyconll.load_from_file(filename)

        for sent in conll:
            for token in sent:
                tag = make_bin(freq_bins, fd.get(language, token.form))
                token.misc["freq"] = (tag,)

        if filename.endswith(".gz"):
            with gzip.open(filename, "wt") as f:
                f.write(conll.conll())
        else:
            with open(filename, "w") as f:
                f.write(conll.conll())


if __name__ == "__main__":
    args = docopt(__doc__)

    log_level = logging.DEBUG if args["--debug"] else logging.INFO
    logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    logzero.loglevel(log_level)

    fd = FrequencyDict(args["--freqfile"])

    try:
        process_submissions(fd)
    except Exception as e:
        log.exception(e)
        raise
