import math
import pytest

# Skip if optional dependency pulls core settings
pytest.importorskip("pydantic_settings")

from core.utils.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    direction_hit_rate,
    confusion_matrix,
    precision_recall_f1,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_regression_metrics_basic():
    y = [1, 2, 3, 4]
    p = [1, 2, 2, 5]
    assert mean_absolute_error(y, p) == (0 + 0 + 1 + 1) / 4
    rmse = root_mean_squared_error(y, p)
    assert rmse == math.sqrt(((0) ** 2 + (0) ** 2 + (1) ** 2 + (1) ** 2) / 4)
    mape = mean_absolute_percentage_error(y, p)
    assert 0 <= mape <= 1


def test_direction_hit_rate():
    y = [1, 2, 3, 2, 4]
    p = [1, 1.5, 2.5, 1.5, 3]
    hr = direction_hit_rate(y, p)
    assert 0.0 <= hr <= 1.0


def test_classification_metrics():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 1, 1, 1, 2, 0]
    labels = [0, 1, 2]
    cm = confusion_matrix(y, p, labels)
    assert len(cm) == 3 and len(cm[0]) == 3
    prf1 = precision_recall_f1(cm)
    assert set(prf1.keys()) == {"per_class", "macro", "micro"}


def test_portfolio_metrics():
    rets = [0.01, -0.02, 0.015, 0.0, 0.005]
    eq = equity_curve(rets, start_equity=1.0)
    assert len(eq) == len(rets) + 1
    mdd = max_drawdown(eq)
    assert -1.0 <= mdd <= 0.0
    s = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    assert isinstance(s, float) and isinstance(so, float)


