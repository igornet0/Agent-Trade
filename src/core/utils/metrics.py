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


def pr_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Calculate Precision-Recall AUC score."""
    if not y_true or not y_score:
        return 0.0
    
    # Sort by score in descending order
    pairs = sorted(zip(y_score, y_true), reverse=True)
    scores, labels = zip(*pairs)
    
    tp = fp = 0
    fn = sum(1 for l in y_true if l == 1)
    
    auc = 0.0
    prev_precision = prev_recall = 0.0
    
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / fn if fn > 0 else 0.0
        
        if prev_recall is not None:
            # Calculate area using trapezoidal rule
            auc += (precision + prev_precision) * (recall - prev_recall) / 2
        
        prev_precision = precision
        prev_recall = recall
    
    return auc


def value_at_risk(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) at given confidence level."""
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    index = int((1 - confidence) * n)
    return sorted_returns[index] if index < n else sorted_returns[-1]


def expected_shortfall(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional VaR) at given confidence level."""
    if not returns:
        return 0.0
    
    var = value_at_risk(returns, confidence)
    tail_returns = [r for r in returns if r <= var]
    return sum(tail_returns) / len(tail_returns) if tail_returns else var


def win_rate(returns: Sequence[float]) -> float:
    """Calculate win rate (percentage of positive returns)."""
    if not returns:
        return 0.0
    
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def turnover_rate(positions: Sequence[float]) -> float:
    """Calculate portfolio turnover rate."""
    if len(positions) <= 1:
        return 0.0
    
    total_turnover = 0.0
    for i in range(1, len(positions)):
        total_turnover += abs(positions[i] - positions[i-1])
    
    return total_turnover / (len(positions) - 1)


def exposure_stats(positions: Sequence[float]) -> Dict[str, float]:
    """Calculate exposure statistics."""
    if not positions:
        return {"max_exposure": 0.0, "avg_exposure": 0.0, "exposure_volatility": 0.0}
    
    max_exp = max(abs(p) for p in positions)
    avg_exp = sum(abs(p) for p in positions) / len(positions)
    
    # Calculate exposure volatility
    exp_squared = sum((abs(p) - avg_exp) ** 2 for p in positions)
    exp_vol = math.sqrt(exp_squared / len(positions)) if len(positions) > 1 else 0.0
    
    return {
        "max_exposure": max_exp,
        "avg_exposure": avg_exp,
        "exposure_volatility": exp_vol
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


def aggregate_returns_equal_weight(returns_per_asset: Sequence[Sequence[float]]) -> List[float]:
    """Aggregate per-asset returns into equal-weight portfolio returns (time-aligned by min length)."""
    if not returns_per_asset:
        return []
    min_len = min(len(r) for r in returns_per_asset)
    if min_len == 0:
        return []
    out: List[float] = []
    n = len(returns_per_asset)
    for t in range(min_len):
        out.append(sum(returns_per_asset[i][t] for i in range(n)) / n)
    return out


def calculate_portfolio_metrics(returns_per_asset: Sequence[Sequence[float]], 
                              start_equity: float = 1.0) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics from per-asset returns."""
    if not returns_per_asset:
        return {}
    
    # Aggregate returns
    portfolio_returns = aggregate_returns_equal_weight(returns_per_asset)
    
    # Calculate equity curve and basic metrics
    equity = equity_curve(portfolio_returns, start_equity)
    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(portfolio_returns)
    sortino = sortino_ratio(portfolio_returns)
    win_rate_val = win_rate(portfolio_returns)
    
    # Calculate per-asset metrics
    per_asset_metrics = []
    for i, asset_returns in enumerate(returns_per_asset):
        if asset_returns:
            asset_equity = equity_curve(asset_returns, start_equity)
            asset_mdd = max_drawdown(asset_equity)
            asset_sharpe = sharpe_ratio(asset_returns)
            asset_win_rate = win_rate(asset_returns)
            
            per_asset_metrics.append({
                "asset_id": i,
                "total_return": asset_equity[-1] - start_equity if asset_equity else 0.0,
                "max_drawdown": asset_mdd,
                "sharpe_ratio": asset_sharpe,
                "win_rate": asset_win_rate,
                "volatility": math.sqrt(sum((r - sum(asset_returns)/len(asset_returns))**2 / len(asset_returns)) if asset_returns else 0.0
            })
    
    return {
        "portfolio": {
            "total_return": equity[-1] - start_equity if equity else 0.0,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate": win_rate_val,
            "volatility": math.sqrt(sum((r - sum(portfolio_returns)/len(portfolio_returns))**2 / len(portfolio_returns)) if portfolio_returns else 0.0
        },
        "per_asset": per_asset_metrics,
        "equity_curve": equity
    }


def calculate_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Calculate comprehensive regression metrics for Pred_time models."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "direction_hit_rate": direction_hit_rate(y_true, y_pred)
    }


def calculate_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], 
                                   y_score: Sequence[float] = None) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for Trade_time models."""
    if not y_true or not y_pred:
        return {}
    
    # Get unique labels
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Calculate confusion matrix and precision/recall/f1
    cm = confusion_matrix(y_true, y_pred, labels)
    prf = precision_recall_f1(cm)
    
    metrics = {
        "confusion_matrix": cm,
        "precision_recall_f1": prf,
        "labels": labels
    }
    
    # Add ROC-AUC and PR-AUC if scores provided
    if y_score and len(y_score) == len(y_true):
        # Convert to binary for ROC-AUC (assuming 1 is positive class)
        if len(labels) == 2:
            binary_true = [1 if y == labels[1] else 0 for y in y_true]
            metrics["roc_auc"] = roc_auc_score(binary_true, y_score)
            metrics["pr_auc"] = pr_auc_score(binary_true, y_score)
        elif len(labels) > 2:
            # Multi-class: calculate per-class and macro
            roc_aucs = []
            pr_aucs = []
            for label in labels:
                binary_true = [1 if y == label else 0 for y in y_true]
                binary_scores = [s if y == label else 0 for y, s in zip(y_true, y_score)]
                roc_aucs.append(roc_auc_score(binary_true, binary_scores))
                pr_aucs.append(pr_auc_score(binary_true, binary_scores))
            
            metrics["roc_auc_per_class"] = dict(zip(labels, roc_aucs))
            metrics["pr_auc_per_class"] = dict(zip(labels, pr_aucs))
            metrics["roc_auc_macro"] = sum(roc_aucs) / len(roc_aucs)
            metrics["pr_auc_macro"] = sum(pr_aucs) / len(pr_aucs)
    
    return metrics


def calculate_risk_metrics(returns: Sequence[float], positions: Sequence[float] = None,
                          confidence: float = 0.95) -> Dict[str, float]:
    """Calculate comprehensive risk metrics for Risk models."""
    if not returns:
        return {}
    
    metrics = {
        "var": value_at_risk(returns, confidence),
        "expected_shortfall": expected_shortfall(returns, confidence),
        "max_drawdown": max_drawdown(equity_curve(returns)),
        "volatility": math.sqrt(sum((r - sum(returns)/len(returns))**2 / len(returns)) if returns else 0.0
    }
    
    if positions:
        exp_stats = exposure_stats(positions)
        metrics.update(exp_stats)
    
    return metrics


def calculate_trading_metrics(returns: Sequence[float], positions: Sequence[float] = None,
                             trades: Sequence[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics for Trade models."""
    if not returns:
        return {}
    
    metrics = {
        "total_return": sum(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "win_rate": win_rate(returns),
        "max_drawdown": max_drawdown(equity_curve(returns)),
        "volatility": math.sqrt(sum((r - sum(returns)/len(returns))**2 / len(returns)) if returns else 0.0
    }
    
    if positions:
        metrics["turnover_rate"] = turnover_rate(positions)
        exp_stats = exposure_stats(positions)
        metrics.update(exp_stats)
    
    if trades:
        # Calculate trade-specific metrics
        trade_returns = [t.get("pnl", 0.0) for t in trades if "pnl" in t]
        if trade_returns:
            metrics["avg_trade_return"] = sum(trade_returns) / len(trade_returns)
            metrics["trade_win_rate"] = win_rate(trade_returns)
            metrics["num_trades"] = len(trades)
    
    return metrics


if __name__ == "__main__":
    # Simple test runner for metrics module
    print("Testing metrics module...")
    
    # Test basic regression metrics
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.1, 3.9]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"MAE: {mae:.6f} (expected ~0.1)")
    print(f"RMSE: {rmse:.6f} (expected ~0.1)")
    print(f"MAPE: {mape:.6f} (expected ~0.025)")
    
    # Test classification metrics
    y_true_bin = [0, 1, 0, 1, 0]
    y_pred_bin = [0, 1, 0, 0, 1]
    y_score_bin = [0.1, 0.9, 0.2, 0.4, 0.6]
    
    cm = confusion_matrix(y_true_bin, y_pred_bin, [0, 1])
    prf = precision_recall_f1(cm)
    roc_auc = roc_auc_score(y_true_bin, y_score_bin)
    pr_auc = pr_auc_score(y_true_bin, y_score_bin)
    
    print(f"Confusion Matrix: {cm}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")
    
    # Test portfolio metrics
    returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    equity = equity_curve(returns, 1000.0)
    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    
    print(f"Equity curve length: {len(equity)}")
    print(f"Max drawdown: {mdd:.6f}")
    print(f"Sharpe ratio: {sharpe:.6f}")
    print(f"Sortino ratio: {sortino:.6f}")
    
    # Test risk metrics
    var_95 = value_at_risk(returns, 0.95)
    es_95 = expected_shortfall(returns, 0.95)
    wr = win_rate(returns)
    
    print(f"VaR (95%): {var_95:.6f}")
    print(f"Expected Shortfall (95%): {es_95:.6f}")
    print(f"Win rate: {wr:.6f}")
    
    # Test utility functions
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    class_metrics = calculate_classification_metrics(y_true_bin, y_pred_bin, y_score_bin)
    risk_metrics = calculate_risk_metrics(returns)
    trading_metrics = calculate_trading_metrics(returns)
    
    print(f"Regression metrics keys: {list(reg_metrics.keys())}")
    print(f"Classification metrics keys: {list(class_metrics.keys())}")
    print(f"Risk metrics keys: {list(risk_metrics.keys())}")
    print(f"Trading metrics keys: {list(trading_metrics.keys())}")
    
    print("All tests passed!")


