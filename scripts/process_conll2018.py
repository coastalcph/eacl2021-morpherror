#!/usr/bin/env python3
"""
Usage: process_conll2018.py [--debug] [--log LOGFILE]

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
from utils.pipeline import FeaturePipeline
from vendor.conll18_ud_eval import load_conllu_file, make_alignment

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = f"{SCRIPTDIR}/../data/conll2018/official-submissions"
OUTDIR = f"{SCRIPTDIR}/../processed/conll2018"


def process_submissions():
    # CoNLL-U column names
    ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

    def add(column, text):
        if column == "_":
            return text
        return f"{column}|{text}"

    pipeline = None
    for goldfile in sorted(glob(f"{DATADIR}/00-gold-standard/*.conllu")):
        treebank = os.path.basename(goldfile)[:-7]
        lang_code = treebank.split("_")[0]

        if lang_code == "no":
            lang_code = "nno" if "nynorsk" in treebank else "nob"

        if pipeline is None or not pipeline.is_lang(lang_code):
            log.info(f"Instantiating pipeline for '{lang_code}'")
            pipeline = FeaturePipeline(lang_code, udpipe=False)

        log.info(f"Processing {treebank} (language: {pipeline.language.name})...")
        # using external loading functions from the official eval script
        conll_gold = load_conllu_file(goldfile)

        for predfile in glob(f"{DATADIR}/**/{treebank}.conllu"):
            team_name = predfile.split("/")[-2]
            if team_name in ("00-gold-standard", "01-blind-input"):
                continue

            log.info(f"Analyzing submission: {team_name}")
            conll_pred = load_conllu_file(predfile)

            alignment = make_alignment(conll_gold, conll_pred)
            lines = []
            sentence_indices = [span.end for span in conll_gold.sentences]

            for words in alignment.matched_words:
                # Using the LAS criterion for "correctness"
                gold = (words.gold_word.parent, words.gold_word.columns[DEPREL])
                pred = (
                    alignment.matched_words_map.get(
                        words.system_word.parent, "NotAligned"
                    )
                    if words.system_word.parent is not None
                    else None,
                    words.system_word.columns[DEPREL],
                )
                tag = "OK" if (gold == pred) else "BAD"
                column = [
                    words.system_word.columns[ID],
                    words.gold_word.columns[
                        FORM
                    ],
                    words.gold_word.columns[LEMMA],
                    words.gold_word.columns[UPOS],
                    words.gold_word.columns[XPOS],
                    words.gold_word.columns[FEATS],
                    words.system_word.columns[HEAD],
                    words.system_word.columns[DEPREL],
                    words.system_word.columns[DEPS],
                    words.system_word.columns[MISC],
                ]
                column = [col if col else "_" for col in column]
                column[MISC] = add(column[MISC], f"tag={tag}")

                lines.append(column)
                if words.gold_word.span.end >= sentence_indices[0]:
                    lines.append([])
                    sentence_indices.pop(0)

            del alignment
            del conll_pred

            # Finally, we create a proper pyconll object out of our lines so we
            # can run the rest of the pipeline
            conll_pred = pyconll.load_from_string(
                "\n".join("\t".join(line) for line in lines)
            )
            conll_pred = pipeline(conll_pred)

            with gzip.open(f"{OUTDIR}/{treebank}.{team_name}.conllu.gz", "wt") as f:
                conll_pred.write(f)


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
