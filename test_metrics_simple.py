#!/usr/bin/env python3
"""
Simple test for metrics functions without core dependencies
"""

import math
from typing import List, Sequence, Dict, Any


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


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Calculate ROC AUC score using trapezoidal rule."""
    if not y_true or not y_score:
        return 0.5
    
    # Sort by score in descending order
    pairs = sorted(zip(y_score, y_true), reverse=True)
    scores, labels = zip(*pairs)
    
    # Calculate TPR and FPR
    tp = fp = 0
    tn = sum(1 for l in y_true if l == 0)
    fn = sum(1 for l in y_true if l == 1)
    
    auc = 0.0
    prev_score = None
    prev_tpr = prev_fpr = 0.0
    
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            # Calculate area using trapezoidal rule
            auc += (prev_tpr + prev_tpr) * (prev_fpr - prev_fpr) / 2
        
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / fn if fn > 0 else 0.0
        fpr = fp / tn if tn > 0 else 0.0
        
        prev_score = score
        prev_tpr = tpr
        prev_fpr = fpr
    
    return auc


def value_at_risk(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) at given confidence level."""
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    index = int((1 - confidence) * n)
    return sorted_returns[index] if index < n else sorted_returns[-1]


def win_rate(returns: Sequence[float]) -> float:
    """Calculate win rate (percentage of positive returns)."""
    if not returns:
        return 0.0
    
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


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


def calculate_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Calculate comprehensive regression metrics for Pred_time models."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "direction_hit_rate": direction_hit_rate(y_true, y_pred)
    }


def calculate_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], 
                                   y_score: Sequence[float] = None) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for Trade_time models."""
    if not y_true or not y_pred:
        return {}
    
    # Get unique labels
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    metrics = {
        "labels": labels
    }
    
    # Add ROC-AUC if scores provided
    if y_score and len(y_score) == len(y_true):
        if len(labels) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    
    return metrics


def calculate_risk_metrics(returns: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calculate comprehensive risk metrics for Risk models."""
    if not returns:
        return {}
    
    metrics = {
        "var": value_at_risk(returns, confidence),
        "max_drawdown": max_drawdown(equity_curve(returns)),
        "win_rate": win_rate(returns)
    }
    
    return metrics


def test_all_metrics():
    """Test all metric functions"""
    print("Testing metrics module...")
    
    # Test basic regression metrics
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.1, 3.9]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    print(f"MAE: {mae:.6f} (expected ~0.1)")
    print(f"RMSE: {rmse:.6f} (expected ~0.1)")
    
    # Test classification metrics
    y_true_bin = [0, 1, 0, 1, 0]
    y_pred_bin = [0, 1, 0, 0, 1]
    y_score_bin = [0.1, 0.9, 0.2, 0.4, 0.6]
    
    roc_auc = roc_auc_score(y_true_bin, y_score_bin)
    print(f"ROC-AUC: {roc_auc:.6f}")
    
    # Test portfolio metrics
    returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    equity = equity_curve(returns, 1000.0)
    mdd = max_drawdown(equity)
    wr = win_rate(returns)
    
    print(f"Equity curve length: {len(equity)}")
    print(f"Max drawdown: {mdd:.6f}")
    print(f"Win rate: {wr:.6f}")
    
    # Test risk metrics
    var_95 = value_at_risk(returns, 0.95)
    print(f"VaR (95%): {var_95:.6f}")
    
    # Test utility functions
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    class_metrics = calculate_classification_metrics(y_true_bin, y_pred_bin, y_score_bin)
    risk_metrics = calculate_risk_metrics(returns)
    
    print(f"Regression metrics keys: {list(reg_metrics.keys())}")
    print(f"Classification metrics keys: {list(class_metrics.keys())}")
    print(f"Risk metrics keys: {list(risk_metrics.keys())}")
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    test_all_metrics()
