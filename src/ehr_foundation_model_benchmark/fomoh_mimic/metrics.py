from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    prevalence: float
    auroc: float
    auprc: float
    pr_lift: float
    brier: float
    ece: float
    ppv_at_100: float
    sensitivity_at_100: float
    calibration_intercept: float
    calibration_slope: float


def _rank_pairs(y_true: list[bool], y_score: list[float]) -> list[tuple[float, bool]]:
    return sorted(zip(y_score, y_true), key=lambda pair: pair[0], reverse=True)


def auroc(y_true: list[bool], y_score: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranked = sorted(zip(y_score, y_true), key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(label for _, label in ranked[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auprc(y_true: list[bool], y_score: list[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return math.nan
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in _rank_pairs(y_true, y_score):
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def brier(y_true: list[bool], y_score: list[float]) -> float:
    return sum((float(label) - score) ** 2 for label, score in zip(y_true, y_score)) / len(y_true)


def ece(y_true: list[bool], y_score: list[float], bins: int = 10) -> float:
    total = len(y_true)
    error = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [
            (label, score)
            for label, score in zip(y_true, y_score)
            if lo <= score < hi or (idx == bins - 1 and score == 1.0)
        ]
        if not bucket:
            continue
        conf = sum(score for _, score in bucket) / len(bucket)
        acc = sum(label for label, _ in bucket) / len(bucket)
        error += len(bucket) / total * abs(acc - conf)
    return error


def workload(y_true: list[bool], y_score: list[float], k: int = 100) -> tuple[float, float]:
    top = _rank_pairs(y_true, y_score)[: min(k, len(y_true))]
    positives = sum(y_true)
    tp = sum(label for _, label in top)
    ppv = tp / len(top) if top else math.nan
    sensitivity = tp / positives if positives else math.nan
    return ppv, sensitivity


def calibration_line(y_true: list[bool], y_score: list[float]) -> tuple[float, float]:
    eps = 1e-6
    logits = [math.log(max(eps, min(1 - eps, score)) / max(eps, 1 - score)) for score in y_score]
    labels = [float(value) for value in y_true]
    x_mean = sum(logits) / len(logits)
    y_mean = sum(labels) / len(labels)
    variance = sum((x - x_mean) ** 2 for x in logits)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(logits, labels)) / variance if variance else math.nan
    intercept = y_mean - (slope * x_mean if not math.isnan(slope) else 0.0)
    return intercept, slope


def summarize_metrics(y_true: list[bool], y_score: list[float]) -> MetricSummary:
    prevalence = sum(y_true) / len(y_true)
    pr_auc = auprc(y_true, y_score)
    ppv, sensitivity = workload(y_true, y_score)
    intercept, slope = calibration_line(y_true, y_score)
    return MetricSummary(
        prevalence=prevalence,
        auroc=auroc(y_true, y_score),
        auprc=pr_auc,
        pr_lift=pr_auc / prevalence if prevalence else math.nan,
        brier=brier(y_true, y_score),
        ece=ece(y_true, y_score),
        ppv_at_100=ppv,
        sensitivity_at_100=sensitivity,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )

