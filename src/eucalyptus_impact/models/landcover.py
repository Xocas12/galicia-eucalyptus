"""Species classification with spatial CV and Olofsson et al. (2014) area estimation.

Counting mapped pixels gives biased areas whenever the classifier errs asymmetrically, which it
always does. The good-practice estimator stratifies a reference sample by *map* class and
re-weights by stratum area. That yields unbiased class areas with confidence intervals, and it is
what any published "eucalyptus hectares in Galicia" figure should rest on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from ..validation.spatial_cv import SpatialBlockKFold


def make_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
    )


@dataclass
class ClassifierReport:
    model: HistGradientBoostingClassifier
    spatial_cv_accuracy: float
    random_cv_accuracy: float
    kappa: float
    f1_per_class: np.ndarray
    confusion: np.ndarray


def train_species_classifier(X, y, groups, n_folds: int = 5, seed: int = 0) -> ClassifierReport:
    """Fit the species classifier and report spatial-block vs random CV skill.

    The gap between random and spatial CV accuracy is itself reported: it is the optimism a
    naive validation would have published.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    labels = np.unique(y)
    pred_sp = np.empty_like(y)
    for tr, te in SpatialBlockKFold(n_folds, seed).split(groups=groups):
        pred_sp[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
    pred_rd = np.empty_like(y)
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        pred_rd[te] = make_classifier(seed).fit(X[tr], y[tr]).predict(X[te])
    model = make_classifier(seed).fit(X, y)
    return ClassifierReport(
        model=model,
        spatial_cv_accuracy=float(accuracy_score(y, pred_sp)),
        random_cv_accuracy=float(accuracy_score(y, pred_rd)),
        kappa=float(cohen_kappa_score(y, pred_sp)),
        f1_per_class=f1_score(y, pred_sp, average=None, labels=labels),
        confusion=confusion_matrix(y, pred_sp, labels=labels),
    )


def stratified_reference_sample(map_labels, per_class: int, rng) -> np.ndarray:
    """Indices of a reference sample stratified by map class (equal allocation, capped)."""
    idx = []
    for c in np.unique(map_labels):
        pool = np.flatnonzero(map_labels == c)
        idx.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return np.concatenate(idx)


def olofsson_area(
    map_labels: np.ndarray,
    ref_map: np.ndarray,
    ref_true: np.ndarray,
    n_classes: int,
    total_area_ha: float,
) -> pd.DataFrame:
    """Stratified area estimator with 95% CIs (Olofsson et al. 2014, eqs. 9-10).

    map_labels: mapped class of every pixel in the population (defines stratum weights W_i).
    ref_map / ref_true: map and reference labels of the stratified reference sample.
    """
    W = np.bincount(map_labels, minlength=n_classes) / len(map_labels)
    n_i = np.bincount(ref_map, minlength=n_classes).astype(float)
    n_ij = np.zeros((n_classes, n_classes))
    np.add.at(n_ij, (ref_map, ref_true), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_i[:, None] > 0, n_ij / n_i[:, None], 0.0)
    p_ij = W[:, None] * frac
    p_j = p_ij.sum(axis=0)
    var = np.zeros(n_classes)
    for j in range(n_classes):
        ok = n_i > 1
        var[j] = np.sum((W[ok] ** 2) * frac[ok, j] * (1 - frac[ok, j]) / (n_i[ok] - 1))
    se = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        users = np.diag(n_ij) / n_i
        producers = np.diag(p_ij) / p_j
    return pd.DataFrame(
        {
            "class": np.arange(n_classes),
            "map_area_ha": W * total_area_ha,
            "est_area_ha": p_j * total_area_ha,
            "ci95_ha": 1.96 * se * total_area_ha,
            "users_accuracy": users,
            "producers_accuracy": producers,
        }
    )
