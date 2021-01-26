#!/usr/bin/env python3
"""
Usage: parse_logfiles.py FILE... [options]

Options:
  --help           Help me!

"""

from docopt import docopt
import logging
import logzero
from logzero import logger as log
import os


LOAD_MSG = "] Loading file: "
SEP = ","


def parse_log(file_, data):
    datafile, stats = None, None
    dummy_acc, dummy_f1 = 0.0, 0.0

    for line in file_:
        line = line.strip()

        if LOAD_MSG in line:
            if datafile is not None:
                if datafile in data:
                    log.warning(f"Data file already processed: {datafile}")
                if "clf_f1" not in stats:
                    log.error(f"Classifier performance not found for {datafile} -- skipping")
                else:
                    stats["diff_f1"] = stats["clf_f1"] - dummy_f1
                    stats["diff_acc"] = stats["clf_acc"] - dummy_acc
                    data[datafile] = stats

            path = os.path.split(line.split(LOAD_MSG)[-1])
            datafile = (os.path.split(path[0])[-1], path[-1])
            stats = {}
            dummy_acc, dummy_f1 = 0.0, 0.0

        elif "Dummy classifier" in line:
            partial, f1 = line.split(", BAD f1 = ")
            dummy_f1 = max(dummy_f1, float(f1))
            dummy_acc = max(dummy_acc, float(partial.split("  acc = ")[-1]))

        elif "Logistic regression classifier" in line or "Random forest classifier" in line:
            if "BAD f1" in line:
                if "clf_f1" in stats:
                    log.error(f"Classifier performance appears twice for {datafile} -- corrupted?")
                partial, f1 = line.split(", BAD f1 = ")
                stats["clf_f1"] = float(f1)
                stats["clf_acc"] = float(partial.split("  acc = ")[-1])
            elif "MCC   = " in line:
                _, mcc = line.split("MCC   = ")
                stats["clf_mcc"] = float(mcc)
            elif "kappa = " in line:
                _, kappa = line.split("kappa = ")
                stats["clf_kappa"] = float(kappa)

    if datafile is not None:
        if datafile in data:
            log.warning(f"Data file already processed: {datafile}")
        if "clf_f1" not in stats:
            log.error(f"No classifier performance not found for {datafile} -- skipping")
        else:
            stats["diff_f1"] = stats["clf_f1"] - dummy_f1
            stats["diff_acc"] = stats["clf_acc"] - dummy_acc
            data[datafile] = stats


if __name__ == "__main__":
    args = docopt(__doc__)
    data = {}

    for filename in args["FILE"]:
        with open(filename, "r") as f:
            parse_log(f, data=data)

    columns = ["dataset", "file", "clf_acc", "clf_f1", "clf_mcc", "clf_kappa", "diff_acc", "diff_f1"]
    print(SEP.join(columns))
    for datafile, stats in data.items():
        a = SEP.join(datafile)
        b = SEP.join(str(stats.get(col, "")) for col in columns[2:])
        print(SEP.join((a, b)))
