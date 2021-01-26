#!/usr/bin/env python3
"""
Usage: process_wmt19_qe.py [--debug] [--log LOGFILE]

Options:
  --log LOGFILE    Additionally write log output to this file.
  --debug          Show debug-level output.
"""

from docopt import docopt
import gzip
import logzero
from logzero import logger as log
import os
import yaml

from utils.pipeline import FeaturePipeline

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = f"{SCRIPTDIR}/../data/wmt19-qe"
OUTDIR = f"{SCRIPTDIR}/../processed/wmt19-qe"


def process_source():
    en_pipe = FeaturePipeline("english")

    for lang in ("de", "ru"):
        datadir = f"{DATADIR}/task1_en-{lang}"
        log.info(f"Loading from {datadir}")

        for subset in ("dev", "train"):
            with open(f"{datadir}/{subset}/{subset}.src", "r") as f:
                lines = f.read()
            with open(f"{datadir}/{subset}/{subset}.source_tags", "r") as f:
                tags = f.readlines()

            log.info(f"Tagging source side of {subset}")
            conll = en_pipe(lines)
            for i, sent_tags in enumerate(tags):
                for j, tag in enumerate(sent_tags.strip().split(" ")):
                    conll[i][j].misc["tag"] = set((tag,))

            outfile = f"{OUTDIR}/task1_en-{lang}.{subset}_src.conll.gz"
            log.info(f"Writing to {outfile}")
            with gzip.open(outfile, "wt") as f:
                conll.write(f)


def process_target():
    for lang in ("de", "ru"):
        datadir = f"{DATADIR}/task1_en-{lang}"
        pipe = FeaturePipeline(lang)
        log.info(f"Loading from {datadir}")

        for subset in ("dev", "train"):
            with open(f"{datadir}/{subset}/{subset}.mt", "r") as f:
                lines = f.read()
            with open(f"{datadir}/{subset}/{subset}.tags", "r") as f:
                tags = f.readlines()

            log.info(f"Tagging target side of {subset}")
            conll = pipe(lines)

            for i, sent_tags in enumerate(tags):
                sent_tags = sent_tags.strip().split(" ")
                proc_tags = [
                    {"tag": set((sent_tags[k],)), "gap_tag": set((sent_tags[k + 1],))}
                    for k in range(1, len(sent_tags), 2)
                ]
                proc_tags[0]["gap_before_tag"] = set((sent_tags[0],))
                for j, tag_dict in enumerate(proc_tags):
                    conll[i][j].misc.update(tag_dict)

            outfile = f"{OUTDIR}/task1_en-{lang}.{subset}_trg.conll.gz"
            log.info(f"Writing to {outfile}")
            with gzip.open(outfile, "wt") as f:
                conll.write(f)


if __name__ == "__main__":
    args = docopt(__doc__)

    import logging

    log_level = logging.DEBUG if args["--debug"] else logging.INFO
    logzero.loglevel(log_level)

    if args["--log"]:
        logzero.logfile(args["--log"])

    os.makedirs(OUTDIR, exist_ok=True)

    try:
        process_source()
        process_target()
    except Exception as e:
        log.exception(e)
        raise
