import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.utils.metrics import (
    mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error,
    direction_hit_rate, confusion_matrix, precision_recall_f1,
    roc_auc_score, pr_auc_score, value_at_risk, expected_shortfall,
    win_rate, turnover_rate, exposure_stats, equity_curve, max_drawdown,
    sharpe_ratio, sortino_ratio, aggregate_returns_equal_weight,
    calculate_portfolio_metrics, calculate_regression_metrics,
    calculate_classification_metrics, calculate_risk_metrics,
    calculate_trading_metrics
)


class TestRegressionMetrics:
    def test_mae(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        assert abs(mean_absolute_error(y_true, y_pred) - 0.1) < 1e-6
    
    def test_rmse(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        assert abs(root_mean_squared_error(y_true, y_pred) - 0.1) < 1e-6
    
    def test_mape(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert 0.02 < mape < 0.03  # ~2.5%
    
    def test_direction_hit_rate(self):
        y_true = [1.0, 2.0, 1.5, 2.5, 2.0, 3.0]
        y_pred = [1.1, 2.1, 1.4, 2.6, 1.9, 3.1]
        hit_rate = direction_hit_rate(y_true, y_pred)
        assert 0.0 <= hit_rate <= 1.0


class TestClassificationMetrics:
    def test_confusion_matrix(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        cm = confusion_matrix(y_true, y_pred, [0, 1])
        assert cm == [[2, 1], [1, 1]]
    
    def test_precision_recall_f1(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        cm = confusion_matrix(y_true, y_pred, [0, 1])
        prf = precision_recall_f1(cm)
        
        assert "macro" in prf
        assert "micro" in prf
        assert "per_class" in prf
        assert len(prf["per_class"]) == 2
    
    def test_roc_auc_score(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        auc = roc_auc_score(y_true, y_score)
        assert 0.0 <= auc <= 1.0
        assert auc > 0.5  # Better than random
    
    def test_pr_auc_score(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        auc = pr_auc_score(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestRiskMetrics:
    def test_value_at_risk(self):
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        var_95 = value_at_risk(returns, 0.95)
        var_99 = value_at_risk(returns, 0.99)
        assert var_95 >= var_99  # Higher confidence = more extreme
        assert var_95 <= 0.0  # Should be negative for 95% confidence
    
    def test_expected_shortfall(self):
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        es = expected_shortfall(returns, 0.95)
        var = value_at_risk(returns, 0.95)
        assert es <= var  # ES should be <= VaR
    
    def test_exposure_stats(self):
        positions = [0.1, 0.2, 0.15, 0.25, 0.3]
        stats = exposure_stats(positions)
        assert "max_exposure" in stats
        assert "avg_exposure" in stats
        assert "exposure_volatility" in stats
        assert stats["max_exposure"] == 0.3


class TestTradingMetrics:
    def test_win_rate(self):
        returns = [-0.01, 0.02, -0.005, 0.015, -0.02, 0.01]
        wr = win_rate(returns)
        assert wr == 0.5  # 3 wins out of 6
    
    def test_turnover_rate(self):
        positions = [0.1, 0.2, 0.15, 0.25, 0.3]
        turnover = turnover_rate(positions)
        assert turnover > 0.0
    
    def test_calculate_trading_metrics(self):
        returns = [-0.01, 0.02, -0.005, 0.015, -0.02, 0.01]
        positions = [0.1, 0.2, 0.15, 0.25, 0.3]
        trades = [{"pnl": 0.01}, {"pnl": -0.005}, {"pnl": 0.02}]
        
        metrics = calculate_trading_metrics(returns, positions, trades)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "turnover_rate" in metrics
        assert "num_trades" in metrics


class TestPortfolioMetrics:
    def test_equity_curve(self):
        returns = [0.01, -0.005, 0.02, -0.01]
        equity = equity_curve(returns, 1000.0)
        assert len(equity) == len(returns) + 1
        assert equity[0] == 1000.0
        assert equity[-1] > 1000.0  # Positive returns
    
    def test_max_drawdown(self):
        equity = [1000, 1100, 1050, 1200, 1150, 1300]
        mdd = max_drawdown(equity)
        assert mdd < 0.0  # Should be negative
        assert mdd >= -0.1  # Max 10% drawdown
    
    def test_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        sharpe = sharpe_ratio(returns)
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        sortino = sortino_ratio(returns)
        assert isinstance(sortino, float)
    
    def test_aggregate_returns_equal_weight(self):
        returns_per_asset = [
            [0.01, 0.02, -0.01],
            [0.015, -0.005, 0.02],
            [0.02, 0.01, 0.005]
        ]
        portfolio_returns = aggregate_returns_equal_weight(returns_per_asset)
        assert len(portfolio_returns) == 3
        assert all(isinstance(r, float) for r in portfolio_returns)
    
    def test_calculate_portfolio_metrics(self):
        returns_per_asset = [
            [0.01, 0.02, -0.01],
            [0.015, -0.005, 0.02],
            [0.02, 0.01, 0.005]
        ]
        metrics = calculate_portfolio_metrics(returns_per_asset, 1000.0)
        
        assert "portfolio" in metrics
        assert "per_asset" in metrics
        assert "equity_curve" in metrics
        
        portfolio = metrics["portfolio"]
        assert "total_return" in portfolio
        assert "max_drawdown" in portfolio
        assert "sharpe_ratio" in portfolio
        
        assert len(metrics["per_asset"]) == 3
        for asset_metrics in metrics["per_asset"]:
            assert "asset_id" in asset_metrics
            assert "total_return" in asset_metrics
            assert "sharpe_ratio" in asset_metrics


class TestUtilityMetrics:
    def test_calculate_regression_metrics(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "direction_hit_rate" in metrics
    
    def test_calculate_classification_metrics(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1]
        y_score = [0.1, 0.9, 0.2, 0.4, 0.6, 0.8]
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_score)
        assert "confusion_matrix" in metrics
        assert "precision_recall_f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
    
    def test_calculate_risk_metrics(self):
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        positions = [0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4]
        
        metrics = calculate_risk_metrics(returns, positions, 0.95)
        assert "var" in metrics
        assert "expected_shortfall" in metrics
        assert "max_drawdown" in metrics
        assert "max_exposure" in metrics


class TestEdgeCases:
    def test_empty_sequences(self):
        assert mean_absolute_error([], []) == 0.0
        assert root_mean_squared_error([], []) == 0.0
        assert roc_auc_score([], []) == 0.5
        assert pr_auc_score([], []) == 0.0
        assert value_at_risk([], 0.95) == 0.0
        assert win_rate([]) == 0.0
    
    def test_single_element(self):
        assert mean_absolute_error([1.0], [1.1]) == 0.1
        assert direction_hit_rate([1.0], [1.1]) == 0.0  # Need at least 2 elements
        assert max_drawdown([1000]) == 0.0
    
    def test_zero_division_handling(self):
        # Test with all zero values
        assert mean_absolute_percentage_error([0.0, 0.0], [0.1, 0.2]) == 0.0
        assert sharpe_ratio([0.0, 0.0, 0.0]) == 0.0
        assert sortino_ratio([0.0, 0.0, 0.0]) == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
