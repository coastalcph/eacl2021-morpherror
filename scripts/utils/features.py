from collections import Counter
from functools import lru_cache
import itertools as it
from logzero import logger as log
import re

from .EdlibWrapper import nice_align


FEAT_JOINER = " && "


@lru_cache(maxsize=4096)
def make_string_features(x, y):
    tags = set()
    pattern = nice_align(x.lower(), y.lower())
    if "|" not in pattern:
        tags.add("edit_full")
    else:
        if pattern[0] != "|":
            tags.add("edit_pre")
        if pattern[-1] != "|":
            tags.add("edit_post")
        if "|" in re.sub("\|+", "", pattern, count=1):
            tags.add("edit_in")

    return tags


def tag_string_features(conll):
    for sent in conll:
        for token in sent:
            if token.form is not None and token.lemma is not None:
                tags = make_string_features(token.form, token.lemma)
                if tags:
                    token.misc["edit"] = tags


def _make_local_features(prefix, token, local_combinations=True):
    # local morph features
    feats = set(
        f"{prefix}:{feat}={','.join(sorted(values))}"
        for feat, values in token.feats.items()
    )

    # combination of all morph features
    if feats and local_combinations:
        feats.add(FEAT_JOINER.join(sorted(feats)))

    # combine POS feature with morph features
    upos = f"{prefix}:upos={token.upos}"
    if local_combinations:
        upos_feats = set(FEAT_JOINER.join([upos, f]) for f in feats)
        upos_feats.add(upos)
        assert len(feats) + 1 == len(upos_feats)
    else:
        upos_feats = set((upos,))

    return feats, upos_feats


def extract_features(
    conll,
    immediate_context=True,
    wide_context=4,
    local_combinations=True,
    control_features=False,
    morph_features=True,
):
    X_feat, y_label = [], []
    if wide_context and isinstance(wide_context, int) and morph_features:
        k = wide_context + 1
        wide_context = list(range(2, k)) + list(range(-2, -k, -1))
    else:
        wide_context = []

    for sent in conll:
        for i, token in enumerate(sent):
            feature_set = set()

            if morph_features:
                # lexical features
                for feat in token.misc.get("lex", []):
                    feature_set.add(feat)

                # string features
                for feat in token.misc.get("edit", []):
                    feature_set.add(feat)

                # morphological features
                local_feats, local_upos_feats = _make_local_features(
                    "tok", token, local_combinations
                )
                feature_set.update(local_feats)
                feature_set.update(local_upos_feats)

            # control features
            if control_features:
                if len(token.form) <= 3:
                    feat = "len1-3"
                elif len(token.form) <= 6:
                    feat = "len4-6"
                elif len(token.form) <= 9:
                    feat = "len7-9"
                else:
                    feat = "len10+"
                feature_set.add(feat)

                for feat in token.misc.get("freq", []):
                    feature_set.add(f"freq={feat}")

            # make immediate context features
            if morph_features and immediate_context:
                for j in (-1, 1):
                    if i + j < 0 or i + j >= len(sent):
                        continue
                    ctx_feats, ctx_upos_feats = _make_local_features(
                        f"tok{j:+d}", sent[i + j]
                    )
                    feature_set.update(ctx_feats)
                    feature_set.update(ctx_upos_feats)

                    # combine with local features
                    for c in it.product(local_upos_feats, ctx_upos_feats):
                        feature_set.add(FEAT_JOINER.join(sorted(c)))

            for j in wide_context:
                if i + j < 0 or i + j >= len(sent):
                    continue
                side = "<<" if j < 0 else ">>"
                ctx_feats, ctx_upos_feats = _make_local_features(
                    f"tok{side}", sent[i + j]
                )

                feature_set.update(ctx_feats)
                feature_set.update(ctx_upos_feats)

                for c in it.product(local_upos_feats, ctx_upos_feats):
                    feature_set.add(FEAT_JOINER.join(sorted(c)))

            X_feat.append(feature_set)
            y_label.append(token.misc["tag"].pop())

    return X_feat, y_label


def prune_features(X_feat, threshold):
    """Prune features whose absolute count is below a given threshold."""
    c = Counter(feat for feature_set in X_feat for feat in feature_set)
    num_before = len(c)
    num_after = num_before - sum(v < threshold for k, v in c.items())

    X_pruned = [
        set(feat for feat in feature_set if c[feat] >= threshold)
        for feature_set in X_feat
    ]

    return X_pruned, num_before, num_after


def max_k_features(X_feat, k):
    """Prune features to retain only the k most frequent ones."""
    c = Counter(feat for feature_set in X_feat for feat in feature_set)
    top_feats = set(feat for feat, _ in c.most_common(k))
    X_pruned = [
        set(feat for feat in feature_set if feat in top_feats) for feature_set in X_feat
    ]

    return X_feat
