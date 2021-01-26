#!/usr/bin/env python3
"""
Usage: process_conll2009.py [--debug] [--log LOGFILE]

Options:
  --log LOGFILE    Additionally write log output to this file.
  --debug          Show debug-level output.
"""

from docopt import docopt
from glob import glob
import gzip
import logzero
from logzero import logger as log
import os

from utils.conll import ConllDataset, ConllFormat
from utils.pipeline import FeaturePipeline

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = f"{SCRIPTDIR}/../data/conll2009"
OUTDIR = f"{SCRIPTDIR}/../processed/conll2009"
DATASETS = ("Catalan", "Chinese", "Czech", "English", "German", "Japanese", "Spanish")


def process_submissions():
    for language in DATASETS:
        log.info(f"Instantiating pipeline for '{language}'")
        pipeline = FeaturePipeline(language)

        for dataset in (language, f"{language}-ood"):
            goldfile = (
                f"{DATADIR}/gold/CoNLL2009-ST-evaluation-{dataset}.14-.PRED_APREDs.txt"
            )
            if not os.path.exists(goldfile):
                continue

            log.info(f"Processing {dataset} (language: {pipeline.language.name})...")
            conll_gold = ConllDataset(columns=["PRED", "APRED"], last_column_multi=True)
            conll_gold.load_from_file(goldfile)
            conll_out = None

            for predfile in glob(
                f"{DATADIR}/eval-data/????/CoNLL2009-ST-evaluation-{dataset}-*.txt"
            ):
                team_name = predfile.split("/")[-2]
                type_name = predfile.split("-")[-1].replace(".txt", "")
                conll_pred = ConllFormat("2009").load_from_file(predfile)

                if not conll_pred.matches_shape(conll_gold):
                    log.error(
                        f"{team_name}-{type_name}/{dataset}: File doesn't seem to line up with gold data!"
                    )
                    continue

                if conll_out is None:
                    conll_out = pipeline(conll_pred.as_horizontal_text())

                log.info(f"Analyzing submission: {team_name}")
                for s_out, s_pred, s_gold in zip(conll_out, conll_pred, conll_gold):
                    for token, pred, gold in zip(s_out, s_pred, s_gold):
                        pred = pred[conll_pred.columns.PRED :]
                        tag = "OK" if pred == gold else "BAD"
                        token.misc["tag"] = (tag,)

                with gzip.open(
                    f"{OUTDIR}/{dataset}.{team_name}-{type_name}.conllu.gz", "wt"
                ) as f:
                    conll_out.write(f)


if __name__ == "__main__":
    args = docopt(__doc__)

    import logging

    log_level = logging.DEBUG if args["--debug"] else logging.INFO
    logzero.loglevel(log_level)

    if args["--log"]:
        logzero.logfile(args["--log"])

    os.makedirs(OUTDIR, exist_ok=True)

    try:
        process_submissions()
    except Exception as e:
        log.exception(e)
        raise
