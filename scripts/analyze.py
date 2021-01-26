#!/usr/bin/env python3
"""
Usage: analyze.py FILE [options]

Analyze feature importance of an FI-based classifier (like random forests).

Options:
  -o, --oob                     Report out-of-bag (OOB) scores. (default: full training set)
  -m, --max-feats NUM           Maximum number of features to extract; features will be
                                pruned down if necessary. [default: -1]
  -n, --control-feats           Also generate non-morphological features as a control.
  -M, --no-morph-feats          Don't generate ANY morphological features.
  -I, --no-immediate-context    Don't generate features for preceding/following tokens.
  -L, --no-local-combinations   Don't generate feature combinations for the same token.
  -w, --wide-context NUM        Number of context tokens to include for wide context
                                features. [default: 4]
  -t, --threshold NUM           Minimum number of occurrences for each feature. [default: 10]
  --method METHOD               Which feature importance method to use; choose between:
                                  * raw (take raw FIs from trained random forest)
                                  * permutation (use permutation importance)
                                  * drop-column (use drop-column importance)
                                  * drop-category (like drop-column, but drops full feature
                                                   categories at a time)
                                  * drop-category-upos (like drop-category, but preserves
                                                        individual POS features)
                                [default: raw]
  --log LOGFILE                 Additionally write log output to this file.

"""

from collections import defaultdict
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
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import sklearn.metrics as skm
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from utils.features import extract_features, prune_features, max_k_features

logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
logzero.loglevel(logging.DEBUG)
SEED = 2300
np.random.seed(SEED)


def calculate_stats(clf, X, y, bad_idx):
    y_pred = clf.predict(X)
    accuracy = sum(y_pred == y) / len(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = skm.precision_recall_fscore_support(y, y_pred, labels=[bad_idx])
    return accuracy, f[0]


def calculate_stats_extended(clf, X, y, bad_idx):
    y_pred = clf.predict(X)
    accuracy = sum(y_pred == y) / len(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = skm.precision_recall_fscore_support(y, y_pred, labels=[bad_idx])
        mcc = skm.matthews_corrcoef(y, y_pred)
        kappa = skm.cohen_kappa_score(y, y_pred)
    return accuracy, f[0], mcc, kappa


def calculate_stats_oob(clf, X, y, bad_idx):
    y_pred = np.argmax(clf.oob_decision_function_, axis=1)
    accuracy = np.mean(y_pred == y, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = skm.precision_recall_fscore_support(y, y_pred, labels=[bad_idx])
        mcc = skm.matthews_corrcoef(y, y_pred)
        kappa = skm.cohen_kappa_score(y, y_pred)
    return accuracy, f[0], mcc, kappa


def get_category(feature, preserve_upos=False):
    if preserve_upos and feature.startswith("tok:upos="):
        return feature
    if "=" in feature:
        return feature.split("=")[0]
    if feature[0] in ("+", "-", "^"):
        return feature[1:]
    if feature.startswith("len"):
        return "len"
    if feature.startswith("edit_"):
        return "edit"
    return feature


def drop_column_importance(
    params, X, y, bad_idx, columns, baselines=None, drop_categories=False, preserve_upos=False
):
    base_acc, base_bf, base_mcc = baselines
    stats_func = (
        calculate_stats_oob if params["oob_score"] else calculate_stats_extended
    )

    if drop_categories:
        drop = defaultdict(list)
        for i, feat in enumerate(columns):
            drop[get_category(feat, preserve_upos=preserve_upos)].append(i)
    else:
        drop = {feat: [i] for i, feat in enumerate(columns)}

    col_importances = {}
    X = X.toarray()  # X is sparse csr_matrix
    for key, drop_columns in drop.items():
        X_drop = np.delete(X, drop_columns, axis=1)
        X_drop = csr_matrix(X_drop)
        clf = RandomForestClassifier(**params).fit(X_drop, y)
        acc, bf, mcc, _ = stats_func(clf, X_drop, y, bad_idx)
        importances = [
            (base_acc - acc),
            (base_bf - bf),
            (base_mcc - mcc),
        ]
        col_importances[key] = importances

    arr_importances = np.zeros((len(columns), 3))
    for i, feat in enumerate(columns):
        if drop_categories:
            key = get_category(feat, preserve_upos=preserve_upos)
        else:
            key = feat
        arr_importances[i] = col_importances[key]

    return arr_importances


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

    threshold = int(args["--threshold"])
    X_f, y_l = extract_features(
        conll,
        immediate_context=not bool(args["--no-immediate-context"]),
        wide_context=int(args["--wide-context"]),
        local_combinations=not bool(args["--no-local-combinations"]),
        control_features=bool(args["--control-feats"]),
        morph_features=not bool(args["--no-morph-feats"]),
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

    clf_params = dict(
        n_estimators=100,
        class_weight="balanced",
        oob_score=bool(args["--oob"]),
        random_state=SEED,
    )
    clf = RandomForestClassifier(**clf_params).fit(X_vec, y_vec)
    stats_func = calculate_stats_oob if args["--oob"] else calculate_stats_extended

    acc, bf, mcc, kappa = stats_func(clf, X_vec, y_vec, bad_idx)
    log.info(f"Random forest classifier:  acc = {acc:.4f}, BAD f1 = {bf:.4f}")
    log.info(f"Random forest classifier:  MCC   = {mcc:.4f}")
    log.info(f"Random forest classifier:  kappa = {kappa:.4f}")

    fi_method = str(args["--method"])
    if fi_method == "permutation":
        pi = permutation_importance(clf, X_vec.toarray(), y_vec, n_repeats=5)
        columns["feature_importance"] = pi["importances_mean"].tolist()
    elif fi_method in ("drop-column", "drop-category", "drop-category-upos"):
        drop_categories = fi_method.startswith("drop-category")
        preserve_upos = fi_method == "drop-category-upos"
        fis = drop_column_importance(
            clf_params,
            X_vec,
            y_vec,
            bad_idx,
            all_feats,
            baselines=(acc, bf, mcc),
            drop_categories=drop_categories,
            preserve_upos=preserve_upos,
        )
        columns["feature_importance"] = fis[:, 1].tolist()
        columns["feature_importance_acc"] = fis[:, 0].tolist()
        columns["feature_importance_mcc"] = fis[:, 2].tolist()
    else:
        if fi_method != "raw":
            log.error(f"Unknown --method '{fi_method}'; falling back to 'raw'")
        columns["feature_importance"] = clf.feature_importances_.tolist()

    # columns["feat_rank"] = ((-1 * clf.feature_importances_).argsort().argsort() + 1).tolist()
    columns["feat_rank"] = [""] * len(columns["feature_importance"])
    scores_with_idx = sorted(
        enumerate(columns["feature_importance"]), key=lambda t: t[1], reverse=True
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
