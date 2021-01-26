import numpy as np
import random
from scipy.stats import spearmanr


def prune_corr_features(X_feat, threshold):
    """Prune features whose correlation score is above a given threshold."""
    feats = list(set(feat for f_set in X_feat for feat in f_set))
    num_before = len(feats)
    step = 2000

    for k in range(5):
        to_keep = set()
        random.shuffle(feats)

        for i in range(0, len(feats), step):
            size = min(step, len(feats) - i)
            x = np.zeros((len(X_feat), size))
            sub_feats = sorted(
                feats[i : i + size], key=lambda f: 30 * f.count(FEAT_JOINER) - len(f)
            )
            for j, x_f in enumerate(sub_feats):
                idx = [x_f in f_set for f_set in X_feat]
                x[idx, j] = 1

            corr, _ = spearmanr(x)
            corr = np.triu(corr, k=1)
            corr = np.any(np.abs(corr) > threshold, axis=0)
            to_keep.update(feat for n, feat in enumerate(sub_feats) if not corr[n])
            log.debug(f"At {i:4d}: eliminated {sum(corr):3d} features")

        feats = list(to_keep)
        log.debug(f"Iteration {k+1}: kept {len(feats)} after pruning")

    return X_pruned, num_before - len(to_prune)
