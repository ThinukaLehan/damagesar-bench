"""Per-fold logistic probes (FR7) on frozen building features.

Pre-registered settings (FYP Master §7, C1 step 5):
* logistic regression, L2, C = 1 (fixed, never tuned), class-balanced weights;
* features standardised with mean and SD from the **training events only** (the held-out
  event never influences the scaler or the probe);
* lbfgs solver, up to 5,000 iterations; a probe that does not converge stops the run.

Folds come from ``data.splits.make_folds``: before the 10 Jan 2027 freeze only development
events are held out and probes train on development + secondary events (never confirmatory);
after the freeze every event is held out in turn. Events with ``units: none`` never train.
Every input and output table is checked with ``contracts``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import EventsConfig
from ..contracts import ContractError, join_events, validate_building_features, validate_fold_scores
from ..data.splits import Fold, make_folds

PROBE_C = 1.0
MAX_ITER = 5000


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeFit:
    held_out: str
    train_events: tuple[str, ...]
    n_train: int
    n_train_damaged: int
    n_iter: int


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=PROBE_C, class_weight="balanced", solver="lbfgs",
                             max_iter=MAX_ITER, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            clf.fit(scaler.transform(x_train), y_train)
        except ConvergenceWarning as ex:
            raise ProbeError(f"probe did not converge in {MAX_ITER} iterations") from ex
    return clf.predict_proba(scaler.transform(x_test))[:, 1], int(np.max(clf.n_iter_))


def run_probes(
    features: pd.DataFrame, events: EventsConfig, today: date | None = None, seed: int = 0,
    folds: list[Fold] | None = None,
) -> tuple[pd.DataFrame, list[ProbeFit]]:
    """Leave-one-event-out probes for every (encoder, pooling) in ``features``.

    Returns the fold-scores table (validated) and one ``ProbeFit`` per fold.
    """
    feats = validate_building_features(features, events, today)
    folds = folds if folds is not None else make_folds(events, today)
    out, fits = [], []
    for (enc, pool), block in features.groupby(["encoder", "pooling"], sort=True):
        present = set(block["event"])
        for fold in folds:
            if fold.held_out not in present:
                continue
            train_events = tuple(e for e in fold.train if e in present)
            missing = set(fold.train) - present
            if missing:
                raise ProbeError(f"{enc}/{pool}: training events {sorted(missing)} have no features; "
                                 "extract them or pass explicit folds")
            train = block[block["event"].isin(train_events)]
            if train["damaged"].isna().any():
                # cannot happen with valid contracts + folds; kept as a second guard
                raise ContractError(f"{enc}/{pool}: unlabelled rows in the training set")
            y = train["damaged"].astype(bool).to_numpy()
            if y.all() or not y.any():
                raise ProbeError(f"{enc}/{pool}: training set for {fold.held_out} has one class")
            test = block[block["event"] == fold.held_out]
            score, n_iter = _fit_predict(train[feats].to_numpy(float), y, test[feats].to_numpy(float), seed)
            part = test[["event", "tile_id", "instance_id", "role", "damaged", "encoder", "pooling"]].copy()
            part["score"] = score
            part["train_events"] = join_events(train_events)
            out.append(part)
            fits.append(ProbeFit(fold.held_out, train_events, len(train), int(y.sum()), n_iter))
    if not out:
        raise ProbeError("no fold could be run: none of the held-out events has features")
    scores = pd.concat(out, ignore_index=True)
    validate_fold_scores(scores, events, today)
    return scores, fits
