#!/usr/bin/env python3
"""
Usage: analyze_ss.py FILE [options]

Analyze feature importance using stability selection.

Options:
  -g, --lambda-grid SPEC        Grid of lambda values (regularization parameter)
                                to iterate over, given as parameters for np.logspace
                                in the format MIN,MAX,NUM. [default: -5,-1,25]
  -i, --max-iter NUM            Maximum number of iterations for stability selection.
                                [default: 100]
  -m, --max-feats NUM           Maximum number of features to extract; features will be
                                pruned down if necessary. [default: -1]
  -n, --control-feats           Also generate non-morphological features as a control.
  -I, --no-immediate-context    Don't generate features for preceding/following tokens.
  -L, --no-local-combinations   Don't generate feature combinations for the same token.
  -w, --wide-context NUM        Number of context tokens to include for wide context
                                features. [default: 4]
  -t, --threshold NUM           Minimum number of occurrences for each feature. [default: 10]
  --log LOGFILE                 Additionally write log output to this file.

"""

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
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from stability_selection import StabilitySelection  # , RandomizedLogisticRegression
from utils.features import extract_features, prune_features, max_k_features
from utils.stats import PatchedRLG as RandomizedLogisticRegression

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
logzero.loglevel(logging.DEBUG)
np.random.seed(2300)


def calculate_stats(clf, X, y, bad_idx):
    y_pred = clf.predict(X)
    accuracy = sum(y_pred == y) / len(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = prfs(y, y_pred, labels=[bad_idx])
    return accuracy, f[0]


def stringify_csv(item):
    if isinstance(item, float):
        return "{:.4f}".format(item)
    if isinstance(item, (set, list, tuple)):
        return "|".join(stringify_csv(i) for i in list(item))
    return str(item)


def output_csv(columns, order=None, file=sys.stdout):
    if order is None:
        order = sorted(columns.keys())

    print("\t".join(order), file=file)
    rank_col = order.index("feat_rank")
    sorted_rows = sorted(
        zip(*[columns[k] for k in order]),
        key=lambda row: row[rank_col] if isinstance(row[rank_col], int) else 9999999,
    )
    for items in sorted_rows:
        print("\t".join(stringify_csv(item) for item in items), file=file)


if __name__ == "__main__":
    args = docopt(__doc__)

    if args["--log"]:
        logzero.logfile(args["--log"])

    log.info(f"Loading file: {args['FILE']}")
    if args["FILE"].endswith(".gz"):
        with gzip.open(args["FILE"], "rt") as f:
            conll = pyconll.load_from_string(f.read())
    else:
        conll = pyconll.load_from_file(args["FILE"])

    lambda_grid_params = [int(x) for x in str(args["--lambda-grid"]).split(",")]
    assert len(lambda_grid_params) == 3, "Invalid lambda grid specification"

    threshold = int(args["--threshold"])
    X_f, y_l = extract_features(
        conll,
        immediate_context=not bool(args["--no-immediate-context"]),
        wide_context=int(args["--wide-context"]),
        local_combinations=not bool(args["--no-local-combinations"]),
        control_features=bool(args["--control-feats"]),
    )
    X_f, num_before, num_after = prune_features(X_f, threshold)
    log.info(
        f"Extracted features: {num_before} before pruning, {num_after} after pruning (threshold = {threshold})"
    )
    max_feats = int(args["--max-feats"])
    if max_feats > 0 and num_after > max_feats:
        X_f, num_after = max_k_features(X_f, max_feats), max_feats
        log.info(f"Frequency cutoff: {num_after} most frequent features retained")

    ## Uncomment this for basic sanity check of calculations
    # for x, y in zip(X_f, y_l):
    #    tag = f"tag={y}"
    #    x.add(tag)

    X_enc = MultiLabelBinarizer(sparse_output=True)
    X_vec = X_enc.fit_transform(X_f)
    y_enc = LabelEncoder()
    y_vec = y_enc.fit_transform(y_l)
    all_feats = X_enc.classes_
    all_labels = y_enc.classes_

    assert "BAD" in all_labels, "Assuming an OK/BAD tagging scheme for now"
    assert len(all_labels) == 2, "Assuming an OK/BAD tagging scheme for now"

    bad_sign = 1 if all_labels[1] == "BAD" else -1
    bad_idx = 1 if all_labels[1] == "BAD" else 0

    ############################################################################
    columns = {"feature_name": all_feats}

    ### Correlation with y_vec
    columns["phi_corrcoef"] = [""] * len(all_feats)
    columns["phi_pvalue"] = [""] * len(all_feats)

    for i in range(len(all_feats)):
        rho, pval = spearmanr(X_vec[:, i].toarray(), y_vec)
        # Multiply by bad_sign so positive correlation ~ correlation with BAD label
        columns["phi_corrcoef"][i] = bad_sign * rho
        columns["phi_pvalue"][i] = pval

    ### Dummy classifier baseline
    clf = DummyClassifier(strategy="stratified").fit(X_vec, y_vec)
    acc, bf = calculate_stats(clf, X_vec, y_vec, bad_idx)
    log.info(f"Dummy classifier (stratified):  acc = {acc:.4f}, BAD f1 = {bf:.4f}")

    clf = DummyClassifier(strategy="most_frequent").fit(X_vec, y_vec)
    acc, bf = calculate_stats(clf, X_vec, y_vec, bad_idx)
    log.info(f"Dummy classifier (majority):  acc = {acc:.4f}, BAD f1 = {bf:.4f}")

    clf = DummyClassifier(strategy="constant", constant=bad_idx).fit(X_vec, y_vec)
    acc, bf = calculate_stats(clf, X_vec, y_vec, bad_idx)
    log.info(f"Dummy classifier (only BAD):  acc = {acc:.4f}, BAD f1 = {bf:.4f}")

    ### Logistic regression
    regression_params = dict(
        solver="liblinear", class_weight="balanced",
        max_iter=int(args["--max-iter"]), C=1.0
    )
    # penalty="l1" is forced by RandomizedLogisticRegression, so we use it here too
    clf = LogisticRegression(penalty="l1", **regression_params).fit(X_vec, y_vec)
    acc, bf = calculate_stats(clf, X_vec, y_vec, bad_idx)
    log.info(f"Logistic regression classifier:  acc = {acc:.4f}, BAD f1 = {bf:.4f}")

    ### Stability selection
    log.info("Performing stability selection...")
    coefs = []
    bfs = []
    n_bootstrap_iterations = 100
    lambda_grid = np.logspace(
        lambda_grid_params[0],
        lambda_grid_params[1],
        num=lambda_grid_params[2]
    )
    progress = tqdm(total=(n_bootstrap_iterations * len(lambda_grid)), unit="runs")

    def callback_fn(clf, X, y):
        coefs.append(clf.coef_)
        _, _, f, _ = prfs(y, clf.predict(X), labels=[bad_idx])
        bfs.append(f[0])
        progress.update()

    selector = StabilitySelection(
        base_estimator=RandomizedLogisticRegression(
            callback_fn=callback_fn, **regression_params
        ),
        lambda_name="C",
        lambda_grid=lambda_grid,
        n_bootstrap_iterations=n_bootstrap_iterations,
        threshold=0.8,  # probably doesn't matter if we're only looking at scores?
        verbose=0,
    )
    selector.fit(X_vec, y_vec)
    progress.close()
    selected_scores = selector.stability_scores_.max(axis=1)
    columns["stability_score"] = list(selected_scores)
    columns["stability_mean_coef"] = bad_sign * np.asarray(coefs).squeeze().mean(axis=0)

    num_feats = len(selector.get_support(indices=True))
    avg_score = sum(selected_scores) / len(selected_scores)
    log.info(f"Selected {num_feats:3d} features, avg. score = {avg_score:.4f}")
    avg_bfs = sum(bfs) / len(bfs)
    log.info(f"Average BAD f1 = {avg_bfs:.4f}")

    columns["feat_rank"] = [""] * len(columns["stability_score"])
    scores_with_idx = sorted(
        enumerate(columns["stability_score"]), key=lambda t: t[1], reverse=True
    )
    i = 1
    correlation_column = "phi_corrcoef"
    for idx, score in scores_with_idx:
        if columns[correlation_column][idx] > 0:  # correlates with BAD label
            columns["feat_rank"][idx] = i
            i += 1

    ### Correlation
    # Using spearmanr in the case of binary labels is identical to calculating
    # phi (or mean square contingency) coefficient
    num_correlate = min(100, i - 1)  # in case there are <100 'BAD' features total
    threshold = 0.8

    if num_correlate > 1:  # edge case, but still
        log.info(f"Correlation analysis for top {num_correlate} 'BAD' features...")
        X_top = np.zeros((len(X_f), num_correlate))
        top_indices = sorted(
            enumerate(columns["feat_rank"]),
            key=lambda t: t[1] if isinstance(t[1], int) else 9999999,
        )[:num_correlate]
        for f_idx, rank_idx in top_indices:
            x_idx = [columns["feature_name"][f_idx] in fset for fset in X_f]
            X_top[x_idx, rank_idx - 1] = 1

        corr, _ = spearmanr(X_top)
        corr = np.triu(corr, k=1)
        above_threshold = (np.abs(corr) > threshold).nonzero()
        columns["x_correlates_with"] = [""] * len(columns["feature_name"])
        for i, j in zip(*above_threshold):
            i_idx = top_indices[i][0]
            j_idx = top_indices[j][0]
            if columns["x_correlates_with"][i_idx]:
                columns["x_correlates_with"][i_idx].append(
                    columns["feature_name"][j_idx]
                )
            else:
                columns["x_correlates_with"][i_idx] = [columns["feature_name"][j_idx]]
            if columns["x_correlates_with"][j_idx]:
                columns["x_correlates_with"][j_idx].append(
                    columns["feature_name"][i_idx]
                )
            else:
                columns["x_correlates_with"][j_idx] = [columns["feature_name"][i_idx]]

        log.info(
            f"Found {len(above_threshold[0])} highly correlated feature pairs (rho > {threshold:.2f})"
        )

    output_csv(columns)
