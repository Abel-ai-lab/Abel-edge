"""Bayesian batch optimization helpers for research workflows."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

DEFAULT_BATCHES = 3
DEFAULT_BATCH_SIZE = 5


def bto_search(
    classifier_cls,
    param_space: dict[str, list],
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_batches: int = DEFAULT_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    classifier_kwargs: dict | None = None,
    score_fn=None,
    val_ret: np.ndarray | None = None,
) -> tuple[dict, float]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required for bto_search(). Install causal-edge with it.") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cls_kwargs = classifier_kwargs or {}
    objective_fn = score_fn or _default_score_fn
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    best_params = None
    best_threshold = 0.5
    best_score = -np.inf

    for _ in range(n_batches):
        trials = [study.ask() for _ in range(batch_size)]
        param_list = []
        for trial in trials:
            params = {name: trial.suggest_categorical(name, values) for name, values in param_space.items()}
            param_list.append(params)

        results = Parallel(n_jobs=-1)(
            delayed(_eval_candidate)(
                classifier_cls,
                params,
                cls_kwargs,
                x_tr,
                y_tr,
                x_val,
                y_val,
                objective_fn,
                val_ret,
            )
            for params in param_list
        )

        for trial, (score, threshold), params in zip(trials, results, param_list):
            study.tell(trial, score)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_params = params

    if best_params is None:
        best_params = {name: values[0] for name, values in param_space.items()}

    return best_params, best_threshold


def sharpe_score_fn(clf, x_val, y_val, val_ret=None):
    raw_prob = clf.predict_proba(x_val)[:, 1]
    signal = np.where(raw_prob > 0.5, 1, 0)
    if val_ret is None:
        return 0.0, 0.5
    pnl = signal * val_ret[: len(signal)]
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    score = float(np.mean(pnl) / std * np.sqrt(252)) if std > 0 else 0.0
    return score, 0.5


def _default_score_fn(clf, x_val, y_val, val_ret=None):
    raw_prob = clf.predict_proba(x_val)[:, 1]
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in np.arange(0.35, 0.66, 0.01):
        prediction = (raw_prob >= threshold).astype(int)
        acc = float(np.mean(prediction == y_val))
        brier = float(np.mean((raw_prob - y_val) ** 2))
        score = acc - 0.10 * brier
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_score, best_threshold


def _eval_candidate(
    classifier_cls,
    params: dict,
    classifier_kwargs: dict,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    score_fn,
    val_ret: np.ndarray | None,
) -> tuple[float, float]:
    try:
        clf = classifier_cls(random_state=42, **classifier_kwargs, **params)
        clf.fit(x_tr, y_tr)
        return score_fn(clf, x_val, y_val, val_ret)
    except Exception:
        return -np.inf, 0.5
