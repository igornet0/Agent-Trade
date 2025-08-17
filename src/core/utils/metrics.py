from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Dict, Any


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        s += abs(float(y_true[i]) - float(y_pred[i]))
    return s / n


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        d = float(y_true[i]) - float(y_pred[i])
        s += d * d
    return math.sqrt(s / n)


def mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-8) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    s = 0.0
    count = 0
    for i in range(n):
        denom = abs(float(y_true[i]))
        if denom < eps:
            continue
        s += abs((float(y_true[i]) - float(y_pred[i])) / denom)
        count += 1
    return (s / count) if count else 0.0


def direction_hit_rate(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n <= 1:
        return 0.0
    hits = 0
    total = 0
    for i in range(1, n):
        dy = float(y_true[i]) - float(y_true[i - 1])
        dp = float(y_pred[i]) - float(y_pred[i - 1])
        sy = 0 if abs(dy) < 1e-12 else (1 if dy > 0 else -1)
        sp = 0 if abs(dp) < 1e-12 else (1 if dp > 0 else -1)
        if sp != 0:  # считаем только когда есть направленный прогноз
            total += 1
            if sp == sy:
                hits += 1
    return (hits / total) if total else 0.0


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> List[List[int]]:
    idx = {label: i for i, label in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    n = min(len(y_true), len(y_pred))
    for i in range(n):
        a = idx.get(int(y_true[i]))
        p = idx.get(int(y_pred[i]))
        if a is None or p is None:
            continue
        m[a][p] += 1
    return m


def precision_recall_f1(cm: List[List[int]]) -> Dict[str, Any]:
    num_classes = len(cm)
    per_class = []
    tp_sum = fp_sum = fn_sum = 0
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    macro_p = sum(pc["precision"] for pc in per_class) / num_classes if num_classes else 0.0
    macro_r = sum(pc["recall"] for pc in per_class) / num_classes if num_classes else 0.0
    macro_f1 = sum(pc["f1"] for pc in per_class) / num_classes if num_classes else 0.0
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
    }


def equity_curve(returns: Sequence[float], start_equity: float = 1.0) -> List[float]:
    eq = [start_equity]
    cur = start_equity
    for r in returns:
        cur *= (1.0 + float(r))
        eq.append(cur)
    return eq


def max_drawdown(equity: Sequence[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, float(v))
        dd = (float(v) / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return mdd


def sharpe_ratio(returns: Sequence[float], eps: float = 1e-12) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    std = math.sqrt(var)
    if std < eps:
        return 0.0
    return mean / std * math.sqrt(len(returns))


def sortino_ratio(returns: Sequence[float], eps: float = 1e-12) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [min(0.0, r) for r in returns]
    var = sum((r) ** 2 for r in downside) / max(len(downside) - 1, 1)
    std = math.sqrt(var)
    if std < eps:
        return 0.0
    return mean / std * math.sqrt(len(returns))


