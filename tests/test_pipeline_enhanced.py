#!/usr/bin/env python3
"""
Test for enhanced pipeline orchestrator with technical indicators and realistic trading
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta


class TestPipelineOrchestrator:
    """Test enhanced pipeline orchestrator functionality"""
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        from src.backend.celery_app.tasks import calculate_sma
        
        prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        sma = calculate_sma(prices, 3)
        
        assert len(sma) == len(prices)
        assert sma[0] is None
        assert sma[1] is None
        assert sma[2] == 2.0  # (1+2+3)/3
        assert sma[3] == 3.0  # (2+3+4)/3
        assert sma[-1] == 9.0  # (8+9+10)/3
    
    def test_calculate_rsi(self):
        """Test Relative Strength Index calculation"""
        from src.backend.celery_app.tasks import calculate_rsi
        
        # Test with increasing prices (should give high RSI)
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        rsi = calculate_rsi(prices, 5)
        
        assert len(rsi) == len(prices)
        assert rsi[0] is None
        assert rsi[4] is None  # First 5 elements are None
        assert rsi[5] is not None
        assert rsi[5] > 70  # High RSI for increasing prices
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        from src.backend.celery_app.tasks import calculate_bollinger_bands
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        upper, middle, lower = calculate_bollinger_bands(prices, 5, 2.0)
        
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
        
        # First 4 elements should be None
        for i in range(4):
            assert upper[i] is None
            assert middle[i] is None
            assert lower[i] is None
        
        # Check that upper > middle > lower
        for i in range(4, len(prices)):
            assert upper[i] > middle[i] > lower[i]
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        from src.backend.celery_app.tasks import calculate_macd
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        macd_line, signal_line, histogram = calculate_macd(prices, 3, 5, 2)
        
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)
        
        # Check that histogram = macd_line - signal_line
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                assert abs(histogram[i] - (macd_line[i] - signal_line[i])) < 1e-10
    
    def test_generate_trading_signals(self):
        """Test trading signal generation"""
        from src.backend.celery_app.tasks import generate_trading_signals
        
        # Mock price data and indicators
        price_data = {'close': [10.0, 11.0, 12.0, 13.0, 14.0]}
        indicators = {
            'sma': [None, 10.5, 11.5, 12.5, 13.5],
            'rsi': [None, 30.0, 50.0, 70.0, 80.0],
            'bb_upper': [None, 12.0, 13.0, 14.0, 15.0],
            'bb_lower': [None, 9.0, 10.0, 11.0, 12.0],
            'macd_line': [None, 0.1, 0.2, 0.3, 0.4],
            'macd_signal': [None, 0.0, 0.1, 0.2, 0.3]
        }
        news_influence = [0.0, 0.1, -0.1, 0.2, -0.2]
        
        signals = generate_trading_signals(price_data, indicators, news_influence)
        
        assert len(signals) == len(price_data['close'])
        assert all(s in [-1, 0, 1] for s in signals)
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        from src.backend.celery_app.tasks import calculate_position_size
        
        # Test with different signal strengths and volatility
        pos_size = calculate_position_size(1, 0.8, 0.05, 1.0)
        assert 0 < pos_size <= 1.0
        
        pos_size = calculate_position_size(-1, 0.6, 0.1, 1.0)
        assert 0 < pos_size <= 1.0
        
        pos_size = calculate_position_size(0, 0.8, 0.05, 1.0)
        assert pos_size == 0.0
    
    def test_calculate_trade_pnl(self):
        """Test trade PnL calculation with costs"""
        from src.backend.celery_app.tasks import calculate_trade_pnl
        
        # Test long position
        pnl = calculate_trade_pnl(100.0, 110.0, 1.0, 1, 0.001, 0.0005)
        expected_pnl = (110.0 - 100.0) / 100.0 - 0.001 - 0.0005
        assert abs(pnl - expected_pnl) < 1e-10
        
        # Test short position
        pnl = calculate_trade_pnl(110.0, 100.0, 1.0, -1, 0.001, 0.0005)
        expected_pnl = (110.0 - 100.0) / 110.0 - 0.001 - 0.0005
        assert abs(pnl - expected_pnl) < 1e-10
        
        # Test no signal
        pnl = calculate_trade_pnl(100.0, 110.0, 1.0, 0, 0.001, 0.0005)
        assert pnl == 0.0
    
    @pytest.mark.asyncio
    async def test_pipeline_backtest_integration(self):
        """Test full pipeline backtest integration"""
        from src.backend.celery_app.tasks import run_pipeline_backtest_task
        
        # Mock configuration
        config = {
            'timeframe': '5m',
            'start': '2025-01-01',
            'end': '2025-01-02',
            'coins': [1, 2],
            'commission': 0.001,
            'slippage': 0.0005,
            'max_position': 1.0,
            'initial_capital': 10000.0,
            'sma_period': 20,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'news_window_bars': 48
        }
        
        # Mock database session and data
        mock_session = AsyncMock()
        mock_session.execute.return_value.all.return_value = []
        
        with patch('src.backend.celery_app.tasks.db_helper.get_session') as mock_db:
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock timeseries data
            mock_ts = Mock()
            mock_ts.id = 1
            mock_session.execute.return_value.scalar.return_value = mock_ts
            
            # Mock OHLCV data
            mock_data = []
            for i in range(100):  # Generate 100 bars of test data
                mock_row = Mock()
                mock_row.datetime = datetime(2025, 1, 1) + timedelta(minutes=5*i)
                mock_row.open = 100.0 + i * 0.1
                mock_row.max = 100.0 + i * 0.1 + 0.5
                mock_row.min = 100.0 + i * 0.1 - 0.5
                mock_row.close = 100.0 + i * 0.1 + 0.2
                mock_row.volume = 1000.0 + i * 10
                mock_data.append(mock_row)
            
            mock_session.execute.return_value.all.return_value = mock_data
            
            # Run the task
            result = run_pipeline_backtest_task(config, pipeline_id=1)
            
            # Check result structure
            assert 'status' in result
            if result['status'] == 'success':
                assert 'metrics' in result
                metrics = result['metrics']
                assert 'portfolio_metrics' in metrics
                assert 'risk_metrics' in metrics
                assert 'trading_metrics' in metrics
                assert 'summary' in metrics
                assert 'artifacts' in metrics


class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    def test_sma_edge_cases(self):
        """Test SMA with edge cases"""
        from src.backend.celery_app.tasks import calculate_sma
        
        # Empty list
        assert calculate_sma([], 5) == []
        
        # List shorter than period
        assert calculate_sma([1.0, 2.0], 5) == [None, None]
        
        # Single element
        assert calculate_sma([1.0], 1) == [1.0]
    
    def test_rsi_edge_cases(self):
        """Test RSI with edge cases"""
        from src.backend.celery_app.tasks import calculate_rsi
        
        # Empty list
        assert calculate_rsi([], 14) == []
        
        # List shorter than period + 1
        assert calculate_rsi([1.0, 2.0], 14) == [None] * 14
        
        # All same prices (no change)
        prices = [10.0] * 20
        rsi = calculate_rsi(prices, 14)
        assert rsi[13] is None  # First 14 elements are None
        assert rsi[14] is not None
    
    def test_bollinger_bands_edge_cases(self):
        """Test Bollinger Bands with edge cases"""
        from src.backend.celery_app.tasks import calculate_bollinger_bands
        
        # Empty list
        upper, middle, lower = calculate_bollinger_bands([], 20)
        assert upper == []
        assert middle == []
        assert lower == []
        
        # List shorter than period
        upper, middle, lower = calculate_bollinger_bands([1.0, 2.0], 20)
        assert all(x is None for x in upper)
        assert all(x is None for x in middle)
        assert all(x is None for x in lower)


class TestTradingLogic:
    """Test trading logic and signal generation"""
    
    def test_signal_thresholds(self):
        """Test signal threshold logic"""
        from src.backend.celery_app.tasks import generate_trading_signals
        
        # Test with strong positive signal
        price_data = {'close': [10.0, 11.0, 12.0]}
        indicators = {
            'sma': [None, 10.5, 11.5],
            'rsi': [None, 25.0, 75.0],  # Oversold then overbought
            'bb_upper': [None, 12.0, 13.0],
            'bb_lower': [None, 9.0, 10.0],
            'macd_line': [None, 0.1, 0.2],
            'macd_signal': [None, 0.0, 0.1]
        }
        news_influence = [0.0, 0.5, -0.5]  # Strong news influence
        
        signals = generate_trading_signals(price_data, indicators, news_influence)
        
        # Should generate buy signal on oversold RSI
        assert signals[1] == 1
        
        # Should generate sell signal on overbought RSI
        assert signals[2] == -1
    
    def test_position_sizing_volatility(self):
        """Test position sizing with different volatility levels"""
        from src.backend.celery_app.tasks import calculate_position_size
        
        # Low volatility should allow larger positions
        pos_size_low_vol = calculate_position_size(1, 0.8, 0.02, 1.0)
        
        # High volatility should reduce positions
        pos_size_high_vol = calculate_position_size(1, 0.8, 0.15, 1.0)
        
        assert pos_size_low_vol > pos_size_high_vol
    
    def test_cost_calculation(self):
        """Test trading cost calculations"""
        from src.backend.celery_app.tasks import calculate_trade_pnl
        
        # Test that costs reduce PnL
        pnl_no_costs = (110.0 - 100.0) / 100.0
        pnl_with_costs = calculate_trade_pnl(100.0, 110.0, 1.0, 1, 0.001, 0.0005)
        
        assert pnl_with_costs < pnl_no_costs
        assert pnl_with_costs == pnl_no_costs - 0.001 - 0.0005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
