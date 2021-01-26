#!/usr/bin/env python3
"""
Usage: process_parseme2018.py [--debug] [--log LOGFILE]

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

import pyconll
from utils.conll import ConllDataset, ConllFormat
from utils.pipeline import FeaturePipeline

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = f"{SCRIPTDIR}/../data/parseme2018/sharedtask-data-master-1.1/1.1"
OUTDIR = f"{SCRIPTDIR}/../processed/parseme2018"


def cut_to_conllu(fields):
    # Blank out fields that can cause parser errors & we don't need here
    return fields[:3] + ["_"] * 7


def fix_id_numbering(conll):
    # some files have invalid ID sequences, and UDPipe chokes on them
    for sentence in conll:
        last_id = None
        renumber = False
        for token in sentence:
            if "-" in token[0]:
                continue
            this_id = float(token[0])
            if last_id is not None and (renumber or this_id <= last_id):
                if not renumber:
                    log.warning("Renumbering a sentence due to bad ID sequence")
                renumber = True
                this_id = round(last_id) + 1
                token[0] = str(this_id)
            last_id = this_id


def find_mwes(sentence):
    mwes = {}
    for i, token in enumerate(sentence):
        if token[-1] in ("", "_", "*"):
            continue
        for expression in token[-1].split(";"):
            if ":" in expression:
                id_, mwe_type = expression.split(":")
            else:
                id_, mwe_type = expression, None

            if id_ in mwes:
                mwes[id_][1].add(i)
                if mwes[id_][0] is None and mwe_type is not None:
                    mwes[id_] = (mwe_type, mwes[id_][1])
            else:
                mwes[id_] = (mwe_type, set([i]))

    return mwes


def find_bad_ids(gold, pred):
    bad = set()
    for gold_type, gold_ids in gold.values():
        # find best-matching mwe in pred
        best = (None, 0)
        for id_, pred_mwe in pred.items():
            pred_type, pred_ids = pred_mwe
            if pred_type == gold_type and len(pred_ids & gold_ids) > best[1]:
                best = (id_, len(pred_ids & gold_ids))
        # match it up
        if best[0] is not None:
            _, pred_ids = pred[best[0]]
            bad.update(gold_ids ^ pred_ids)
            del pred[best[0]]
    # anything not matched up yet?
    for _, pred_ids in pred.values():
        bad.update(pred_ids)

    return bad


def process_submissions():
    for golddir in glob(f"{DATADIR}/??"):
        language = golddir[-2:].lower()
        if language == "ar":
            continue  # Arabic is not included due to lack of open license

        log.info(f"Instantiating pipeline for '{language}'")
        pipeline = FeaturePipeline(language, udpipe=True)

        conll_gold = ConllFormat("PARSEME").load_from_file(f"{golddir}/test.cupt")
        fix_id_numbering(conll_gold)
        conll_out = pyconll.load_from_string(
            conll_gold.to_string(col_transform=cut_to_conllu)
        )
        conll_out = pipeline(conll_out)

        for dataset in glob(
            f"{DATADIR}/system-results/**/{language.upper()}/test.system.cupt"
        ):
            system_name = dataset.split("/")[-3]
            log.info(
                f"Processing {system_name} (language: {pipeline.language.name})..."
            )

            conll_pred = ConllFormat("PARSEME").load_from_file(dataset)

            for gold, pred, out in zip(conll_gold, conll_pred, conll_out):
                gold_mwes = find_mwes(gold)
                pred_mwes = find_mwes(pred)
                errors = find_bad_ids(gold_mwes, pred_mwes)

                for i, token in enumerate(out):
                    tag = "BAD" if i in errors else "OK"
                    token.misc["tag"] = (tag,)

            with gzip.open(f"{OUTDIR}/{language}.{system_name}.conllu.gz", "wt") as f:
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
