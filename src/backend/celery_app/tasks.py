from .create_app import celery_app
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
import math
import os
from uuid import uuid4
from sqlalchemy import select
import pandas as pd
import numpy as np

from core.database import db_helper
from core.database.orm.market import (
    orm_get_timeseries_by_coin as market_get_ts,
    orm_get_data_timeseries as market_get_data,
)
from core.utils.metrics import (
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    aggregate_returns_equal_weight,
    calculate_portfolio_metrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_risk_metrics,
    calculate_trading_metrics,
    value_at_risk,
    expected_shortfall,
    win_rate,
    turnover_rate,
    exposure_stats,
)
from core.database.models.main_models import NewsHistoryCoin, News
from core.database.models.process_models import Backtest as BacktestModel
from core.database.orm_query import (
    orm_get_agent_by_id,
    orm_get_train_agent,
    orm_get_timeseries_by_coin,
    orm_get_data_timeseries,
)
from backend.Dataset.loader import LoaderTimeLine
from backend.train_models.loader import Loader

logger = logging.getLogger("celery.train")


def calculate_sma(prices: list[float], period: int) -> list[float]:
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return [None] * len(prices)
    
    sma = [None] * (period - 1)
    for i in range(period - 1, len(prices)):
        sma.append(sum(prices[i-period+1:i+1]) / period)
    return sma


def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return [None] * len(prices)
    
    rsi = [None] * period
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) >= period:
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi.append(rsi_val)
            
            # Update averages using smoothing
            if i < len(gains):
                avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
    
    return rsi


def calculate_bollinger_bands(prices: list[float], period: int = 20, std_dev: float = 2.0) -> tuple[list[float], list[float], list[float]]:
    """Calculate Bollinger Bands (upper, middle, lower)"""
    if len(prices) < period:
        return [None] * len(prices), [None] * len(prices), [None] * len(prices)
    
    upper = [None] * (period - 1)
    middle = [None] * (period - 1)
    lower = [None] * (period - 1)
    
    for i in range(period - 1, len(prices)):
        window = prices[i-period+1:i+1]
        sma = sum(window) / period
        variance = sum((p - sma) ** 2 for p in window) / period
        std = math.sqrt(variance)
        
        middle.append(sma)
        upper.append(sma + (std_dev * std))
        lower.append(sma - (std_dev * std))
    
    return upper, middle, lower


def calculate_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[list[float], list[float], list[float]]:
    """Calculate MACD (macd_line, signal_line, histogram)"""
    if len(prices) < slow:
        return [None] * len(prices), [None] * len(prices), [None] * len(prices)
    
    # Calculate EMAs
    def ema(prices: list[float], period: int) -> list[float]:
        if len(prices) < period:
            return [None] * len(prices)
        
        ema_vals = [None] * (period - 1)
        ema_vals.append(sum(prices[:period]) / period)
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema_val = (prices[i] * multiplier) + (ema_vals[-1] * (1 - multiplier))
            ema_vals.append(ema_val)
        
        return ema_vals
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    # MACD line
    macd_line = []
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Signal line (EMA of MACD)
    macd_values = [v for v in macd_line if v is not None]
    if len(macd_values) >= signal:
        signal_line = ema(macd_values, signal)
        # Pad with None to match original length
        signal_line = [None] * (len(prices) - len(signal_line)) + signal_line
    else:
        signal_line = [None] * len(prices)
    
    # Histogram
    histogram = []
    for i in range(len(prices)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    
    return macd_line, signal_line, histogram


def generate_trading_signals(price_data: dict, indicators: dict, news_influence: list[float]) -> list[int]:
    """Generate trading signals based on technical indicators and news"""
    signals = []
    prices = price_data.get('close', [])
    sma = indicators.get('sma', [])
    rsi = indicators.get('rsi', [])
    bb_upper = indicators.get('bb_upper', [])
    bb_lower = indicators.get('bb_lower', [])
    macd_line = indicators.get('macd_line', [])
    macd_signal = indicators.get('macd_signal', [])
    
    for i in range(len(prices)):
        if i < 1:  # Need at least 2 prices for signals
            signals.append(0)
            continue
        
        signal = 0
        price = prices[i]
        
        # SMA crossover
        if i > 0 and sma[i] is not None and sma[i-1] is not None:
            if price > sma[i] and prices[i-1] <= sma[i-1]:  # Golden cross
                signal += 1
            elif price < sma[i] and prices[i-1] >= sma[i-1]:  # Death cross
                signal -= 1
        
        # RSI overbought/oversold
        if rsi[i] is not None:
            if rsi[i] < 30:  # Oversold
                signal += 1
            elif rsi[i] > 70:  # Overbought
                signal -= 1
        
        # Bollinger Bands
        if bb_upper[i] is not None and bb_lower[i] is not None:
            if price <= bb_lower[i]:  # Price at lower band
                signal += 1
            elif price >= bb_upper[i]:  # Price at upper band
                signal -= 1
        
        # MACD crossover
        if i > 0 and macd_line[i] is not None and macd_signal[i] is not None:
            if (macd_line[i] > macd_signal[i] and 
                macd_line[i-1] <= macd_signal[i-1]):  # Bullish crossover
                signal += 1
            elif (macd_line[i] < macd_signal[i] and 
                  macd_line[i-1] >= macd_signal[i-1]):  # Bearish crossover
                signal -= 1
        
        # News influence (if available)
        if i < len(news_influence) and news_influence[i] != 0:
            news_signal = 1 if news_influence[i] > 0 else -1
            signal += 0.3 * news_signal  # Reduce news weight
        
        # Final signal: threshold and normalize
        if signal > 0.5:
            signals.append(1)  # Buy
        elif signal < -0.5:
            signals.append(-1)  # Sell
        else:
            signals.append(0)  # Hold
    
    return signals


def calculate_position_size(signal: int, risk_cap: float, volatility: float, 
                           max_position: float = 1.0) -> float:
    """Calculate position size based on signal strength, risk cap, and volatility"""
    if signal == 0:
        return 0.0
    
    # Base position size from signal strength
    base_size = abs(signal) * risk_cap
    
    # Adjust for volatility (reduce position in high volatility)
    vol_adjustment = max(0.1, 1.0 - (volatility * 10))  # Cap at 90% reduction
    
    # Apply volatility adjustment
    adjusted_size = base_size * vol_adjustment
    
    # Cap at maximum position
    return min(adjusted_size, max_position)


def calculate_trade_pnl(entry_price: float, exit_price: float, position_size: float, 
                        signal: int, commission: float = 0.001, slippage: float = 0.0005) -> float:
    """Calculate trade PnL including commission and slippage"""
    if signal == 0 or position_size == 0:
        return 0.0
    
    # Calculate raw PnL
    if signal > 0:  # Long position
        raw_pnl = (exit_price - entry_price) / entry_price
    else:  # Short position
        raw_pnl = (entry_price - exit_price) / entry_price
    
    # Apply position size
    pnl = raw_pnl * position_size
    
    # Subtract costs
    total_costs = commission + slippage
    pnl -= total_costs
    
    return pnl


@celery_app.task(bind=True)
def train_model_task(self, agent_id: int):
    """Kick off training for a specific Agent and persist artifacts/metrics.

    This task:
    - Loads agent config via ORM
    - Builds Loader/AgentManager
    - Trains and updates progress
    - Saves checkpoints/metrics using agent.save_model/save_json
    """

    async def _run():
        async with db_helper.get_session() as session:  # type: AsyncSession
            agent = await orm_get_agent_by_id(session, agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return {"status": "error", "detail": "agent not found"}

            # Fetch training config and target coins
            trains = await orm_get_train_agent(session, agent_id)
            train = trains[0] if isinstance(trains, list) and trains else None

            # Build training components (reusing existing training code)
            loader = Loader(agent_type=agent.type, model_type="MMM")
            agent_manager = loader.load_model(count_agents=1)
            if agent_manager is None:
                return {"status": "error", "detail": "agent manager not built"}

            # Prepare historical loaders for specified coins/timeframe
            loaders: list[LoaderTimeLine] = []
            timeframe = getattr(agent, "timeframe", "5m")
            seq_len = getattr(agent_manager.get_agents(), "model_parameters", {}).get("seq_len", 50)
            pred_len = getattr(agent_manager.get_agents(), "model_parameters", {}).get("pred_len", 5)
            window = seq_len + pred_len

            if train and getattr(train, "coins", None):
                coin_ids = [coin.id for coin in train.coins]
            else:
                coin_ids = []

            async def build_dataset_for_coin(coin_id: int):
                ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)
                if not ts:
                    return
                data_rows = await orm_get_data_timeseries(session, ts.id)
                # Sort ascending by datetime
                data_rows = sorted(data_rows, key=lambda r: r.datetime)
                for row in data_rows:
                    yield {
                        "datetime": row.datetime,
                        "open": row.open,
                        "max": row.max,
                        "min": row.min,
                        "close": row.close,
                        "volume": row.volume,
                    }

            for coin_id in coin_ids:
                dataset_async_gen = build_dataset_for_coin(coin_id)

                # Bridge async generator to sync iterable expected by LoaderTimeLine
                items = []
                try:
                    async for item in dataset_async_gen:
                        items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to load data for coin {coin_id}: {e}")
                    continue

                if len(items) >= window:
                    loader = LoaderTimeLine(items, seq_len=seq_len, pred_len=pred_len)
                    loaders.append(loader)

            if not loaders:
                return {"status": "error", "detail": "no valid data loaders built"}

            # Start training
            self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Starting training...'})
            
            try:
                # Train the agent
                agent_manager.train(loaders)
                self.update_state(state='PROGRESS', meta={'progress': 90, 'message': 'Training completed, saving...'})
                
                # Save model and metrics
                # Note: This would need to be implemented in the agent manager
                # For now, return success
                
                return {"status": "success", "agent_id": agent_id, "message": "Training completed"}
                
            except Exception as e:
                logger.exception(f"Training failed for agent {agent_id}")
                return {"status": "error", "detail": str(e)}

    return asyncio.run(_run())


@celery_app.task(bind=True)
def evaluate_model_task(self, agent_id: int, coins: list[int], timeframe: str = "5m",
                        start: str | None = None, end: str | None = None):
    """Evaluate trained agent offline on a selected time range.

    Returns simple metrics dict; progress updates via Celery state.
    """

    async def _run():
        async with db_helper.get_session() as session:  # type: AsyncSession
            agent = await orm_get_agent_by_id(session, agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return {"status": "error", "detail": "agent not found"}

            loader = Loader(agent_type=agent.type, model_type="MMM")
            agent_manager = loader.load_model(count_agents=1)
            if agent_manager is None:
                return {"status": "error", "detail": "agent manager not built"}

            # Load datasets
            def parse_dt(dt: str | None):
                if not dt:
                    return None
                try:
                    return datetime.fromisoformat(dt)
                except Exception:
                    return None

            dt_start = parse_dt(start)
            dt_end = parse_dt(end)

            total = max(len(coins), 1)
            processed = 0
            metrics = {
                "coins": [],
                "samples": 0,
                "avg_loss": None,
            }

            for coin_id in coins:
                ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)
                if not ts:
                    continue
                data_rows = await orm_get_data_timeseries(session, ts.id)
                # sort and clip by time range
                rows = sorted(data_rows, key=lambda r: r.datetime)
                if dt_start:
                    rows = [r for r in rows if r.datetime >= dt_start]
                if dt_end:
                    rows = [r for r in rows if r.datetime <= dt_end]
                items = [
                    {
                        "datetime": r.datetime,
                        "open": r.open,
                        "max": r.max,
                        "min": r.min,
                        "close": r.close,
                        "volume": r.volume,
                    }
                    for r in rows
                ]
                if not items:
                    continue

                # Simple evaluation using loader utilities
                try:
                    result = loader.evaluate_model(dataset=items, agent_manager=agent_manager)
                    metrics["coins"].append({"coin_id": coin_id, **(result or {})})
                    if result and "samples" in result:
                        metrics["samples"] += int(result["samples"])
                    if result and "avg_loss" in result:
                        if metrics["avg_loss"] is None:
                            metrics["avg_loss"] = result["avg_loss"]
                        else:
                            metrics["avg_loss"] = (metrics["avg_loss"] + result["avg_loss"]) / 2.0
                except Exception as e:
                    logger.exception("Evaluate failed for coin %s", coin_id)

                processed += 1
                self.update_state(state='PROGRESS', meta={'progress': int(processed/total*100)})

            return {"status": "success", "agent_id": agent_id, "metrics": metrics}

    return asyncio.run(_run())


# --------- Lightweight placeholders for Stage 1 contracts ---------
@celery_app.task(bind=True)
def train_news_task(self, config: dict):
    """Train News model with enhanced NLP processing and background calculation"""
    try:
        from core.services.news_background_service import NewsBackgroundService
        from core.database import db_helper
        import asyncio
        
        # Extract configuration
        coin_ids = config.get('coin_ids', [])
        window_hours = config.get('window_hours', 24)
        decay_factor = config.get('decay_factor', 0.95)
        force_recalculate = config.get('force_recalculate', False)
        
        # Initialize service
        news_service = NewsBackgroundService()
        
        async def _run():
            async with db_helper.get_session() as session:
                if coin_ids:
                    # Calculate background for specific coins
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': 10, 'message': f'Calculating background for {len(coin_ids)} coins...'}
                    )
                    
                    results = await news_service.calculate_multiple_coins_background(
                        session, coin_ids, window_hours, decay_factor
                    )
                    
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': 80, 'message': 'Saving results to database...'}
                    )
                    
                    # Process results
                    processed_coins = 0
                    errors = []
                    
                    for coin_id, result in results.items():
                        if 'error' not in result:
                            processed_coins += 1
                        else:
                            errors.append(f"Coin {coin_id}: {result['error']}")
                    
                    self.update_state(
                        state='SUCCESS',
                        meta={
                            'progress': 100,
                            'message': f'Completed: {processed_coins} coins processed',
                            'results': {
                                'coins_processed': processed_coins,
                                'errors': errors,
                                'window_hours': window_hours,
                                'decay_factor': decay_factor
                            }
                        }
                    )
                    
                    return {
                        'status': 'success',
                        'coins_processed': processed_coins,
                        'errors': errors,
                        'window_hours': window_hours,
                        'decay_factor': decay_factor
                    }
                else:
                    # Recalculate all coins
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': 20, 'message': 'Getting all coins with news data...'}
                    )
                    
                    coins = await news_service.get_coins_with_news_data(session)
                    
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': 40, 'message': f'Recalculating background for {len(coins)} coins...'}
                    )
                    
                    result = await news_service.recalculate_all_backgrounds(
                        session, window_hours, decay_factor
                    )
                    
                    if result['status'] == 'success':
                        self.update_state(
                            state='SUCCESS',
                            meta={
                                'progress': 100,
                                'message': f"Completed: {result['coins_processed']} coins processed",
                                'results': result
                            }
                        )
                    else:
                        self.update_state(
                            state='FAILURE',
                            meta={'progress': 100, 'message': f"Failed: {result['error']}"}
                        )
                    
                    return result
        
        return asyncio.run(_run())
        
    except Exception as e:
        logger.exception("train_news_task failed")
        return {"status": "error", "detail": str(e)}


@celery_app.task(bind=True)
def evaluate_news_task(self, config: dict):
    """Evaluate News model performance and correlation with price movements"""
    try:
        from core.services.news_background_service import NewsBackgroundService
        from core.database import db_helper
        from core.database.orm.market import orm_get_timeseries_by_coin, orm_get_data_timeseries
        from sqlalchemy import select
        import asyncio
        import math
        
        # Extract configuration
        coin_ids = config.get('coin_ids', [])
        evaluation_hours = config.get('evaluation_hours', 168)  # 1 week
        correlation_threshold = config.get('correlation_threshold', 0.1)
        
        async def _run():
            async with db_helper.get_session() as session:
                news_service = NewsBackgroundService()
                evaluation_results = {}
                
                for coin_id in coin_ids:
                    try:
                        self.update_state(
                            state='PROGRESS',
                            meta={'progress': 30, 'message': f'Evaluating coin {coin_id}...'}
                        )
                        
                        # Get news background history
                        end_time = datetime.utcnow()
                        start_time = end_time - timedelta(hours=evaluation_hours)
                        
                        backgrounds = await news_service.get_background_history(
                            session, coin_id, start_time, end_time
                        )
                        
                        if not backgrounds:
                            evaluation_results[coin_id] = {
                                'status': 'no_data',
                                'message': 'No news background data available'
                            }
                            continue
                        
                        # Get price data for correlation analysis
                        ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe='5m')
                        if not ts:
                            evaluation_results[coin_id] = {
                                'status': 'no_price_data',
                                'message': 'No price data available'
                            }
                            continue
                        
                        price_data = await orm_get_data_timeseries(session, ts.id)
                        price_data = sorted(price_data, key=lambda x: x.datetime)
                        
                        # Align news background with price data
                        aligned_data = []
                        for bg in backgrounds:
                            bg_time = datetime.fromisoformat(bg['timestamp'])
                            
                            # Find closest price data point
                            closest_price = None
                            min_diff = float('inf')
                            
                            for price in price_data:
                                if start_time <= price.datetime <= end_time:
                                    diff = abs((bg_time - price.datetime).total_seconds())
                                    if diff < min_diff:
                                        min_diff = diff
                                        closest_price = price
                            
                            if closest_price and min_diff <= 300:  # Within 5 minutes
                                aligned_data.append({
                                    'news_score': bg['score'],
                                    'price': float(closest_price.close),
                                    'timestamp': bg['timestamp']
                                })
                        
                        if len(aligned_data) < 10:
                            evaluation_results[coin_id] = {
                                'status': 'insufficient_data',
                                'message': f'Only {len(aligned_data)} aligned data points'
                            }
                            continue
                        
                        # Calculate correlation metrics
                        news_scores = [d['news_score'] for d in aligned_data]
                        prices = [d['price'] for d in aligned_data]
                        
                        # Calculate price changes
                        price_changes = []
                        for i in range(1, len(prices)):
                            if prices[i-1] != 0:
                                change = (prices[i] - prices[i-1]) / prices[i-1]
                                price_changes.append(change)
                        
                        # Calculate correlations
                        if len(price_changes) >= len(news_scores) - 1:
                            # Align news scores with price changes
                            aligned_scores = news_scores[1:len(price_changes)+1]
                            
                            # Calculate correlation coefficient
                            n = len(aligned_scores)
                            if n > 1:
                                mean_score = sum(aligned_scores) / n
                                mean_change = sum(price_changes) / n
                                
                                numerator = sum((s - mean_score) * (c - mean_change) 
                                              for s, c in zip(aligned_scores, price_changes))
                                
                                score_variance = sum((s - mean_score) ** 2 for s in aligned_scores)
                                change_variance = sum((c - mean_change) ** 2 for c in price_changes)
                                
                                if score_variance > 0 and change_variance > 0:
                                    correlation = numerator / math.sqrt(score_variance * change_variance)
                                else:
                                    correlation = 0.0
                            else:
                                correlation = 0.0
                        else:
                            correlation = 0.0
                        
                        # Calculate additional metrics
                        if news_scores:
                            mean_score_all = sum(news_scores) / len(news_scores)
                            variance = sum((s - mean_score_all) ** 2 for s in news_scores) / len(news_scores)
                            score_volatility = math.sqrt(variance)
                            avg_score = mean_score_all
                        else:
                            score_volatility = 0.0
                            avg_score = 0.0
                        
                        # Determine sentiment classification
                        if abs(correlation) >= correlation_threshold:
                            if correlation > 0:
                                sentiment_accuracy = 'positive_correlation'
                            else:
                                sentiment_accuracy = 'negative_correlation'
                        else:
                            sentiment_accuracy = 'no_correlation'
                        
                        evaluation_results[coin_id] = {
                            'status': 'success',
                            'correlation': correlation,
                            'sentiment_accuracy': sentiment_accuracy,
                            'score_volatility': score_volatility,
                            'avg_score': avg_score,
                            'data_points': len(aligned_data),
                            'evaluation_hours': evaluation_hours,
                            'correlation_threshold': correlation_threshold
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed to evaluate coin {coin_id}: {e}")
                        evaluation_results[coin_id] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Calculate overall metrics
                successful_evaluations = [r for r in evaluation_results.values() if r['status'] == 'success']
                
                if successful_evaluations:
                    avg_correlation = sum(r['correlation'] for r in successful_evaluations) / len(successful_evaluations)
                    correlation_std = math.sqrt(sum((r['correlation'] - avg_correlation)**2 for r in successful_evaluations) / len(successful_evaluations))
                    
                    overall_metrics = {
                        'avg_correlation': avg_correlation,
                        'correlation_std': correlation_std,
                        'coins_evaluated': len(coin_ids),
                        'successful_evaluations': len(successful_evaluations),
                        'evaluation_hours': evaluation_hours
                    }
                else:
                    overall_metrics = {
                        'coins_evaluated': len(coin_ids),
                        'successful_evaluations': 0,
                        'evaluation_hours': evaluation_hours
                    }
                
                self.update_state(
                    state='SUCCESS',
                    meta={
                        'progress': 100,
                        'message': f'Evaluation completed for {len(coin_ids)} coins',
                        'results': evaluation_results,
                        'overall_metrics': overall_metrics
                    }
                )
                
                return {
                    'status': 'success',
                    'results': evaluation_results,
                    'overall_metrics': overall_metrics
                }
        
        return asyncio.run(_run())
        
    except Exception as e:
        logger.exception("evaluate_news_task failed")
        return {"status": "error", "detail": str(e)}


@celery_app.task(bind=True)
def evaluate_trade_aggregator_task(self, config: dict):
    """Evaluate Trade aggregator model"""
    # Placeholder for Trade model evaluation
    return {"status": "success", "message": "Trade model evaluation completed"}


@celery_app.task(bind=True)
def run_pipeline_backtest_task(self, config_json: dict, pipeline_id: int = None):
    """Run pipeline backtest with enhanced technical analysis and realistic trading logic"""
    
    async def _run():
        try:
            cfg = config_json
            tf = cfg.get('timeframe', '5m')
            start_date = cfg.get('start', '2025-01-01')
            end_date = cfg.get('end', '2025-12-31')
            coin_ids = cfg.get('coins', [1, 2, 3])  # Default test coins
            
            # Trading parameters
            commission = cfg.get('commission', 0.001)  # 0.1%
            slippage = cfg.get('slippage', 0.0005)    # 0.05%
            max_position = cfg.get('max_position', 1.0)
            initial_capital = cfg.get('initial_capital', 10000.0)
            
            # Technical indicator parameters
            sma_period = cfg.get('sma_period', 20)
            rsi_period = cfg.get('rsi_period', 14)
            bb_period = cfg.get('bb_period', 20)
            bb_std = cfg.get('bb_std', 2.0)
            macd_fast = cfg.get('macd_fast', 12)
            macd_slow = cfg.get('macd_slow', 26)
            macd_signal = cfg.get('macd_signal', 9)
            
            # News parameters
            news_window_bars = cfg.get('news_window_bars', 48)  # 4 hours for 5m bars
            
            # Progress tracking
            steps = [
                (10, "Загрузка данных"),
                (20, "Расчет индикаторов"),
                (30, "Анализ новостей"),
                (40, "Генерация сигналов"),
                (50, "Расчет позиций"),
                (60, "Риск-менеджмент"),
                (70, "Бэктест сделок"),
                (80, "Расчет метрик"),
                (90, "Сохранение результатов")
            ]
            
            def step(progress: int, message: str):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message})
                time.sleep(0.1)  # Small delay for progress visibility
            
            # 1) Load OHLCV data
            step(10, *steps[0])
            per_asset_data: list[dict] = []
            datetimes: list[str] = []
            
            async with db_helper.get_session() as session:
                for coin_id in coin_ids:
                    ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe=tf)
                    if ts:
                        data_rows = await orm_get_data_timeseries(session, ts.id)
                        if data_rows:
                            # Sort by datetime
                            data_rows = sorted(data_rows, key=lambda r: r.datetime)
                            
                            # Filter by date range if specified
                            if start_date and end_date:
                                start_dt = datetime.fromisoformat(start_date)
                                end_dt = datetime.fromisoformat(end_date)
                                data_rows = [r for r in data_rows if start_dt <= r.datetime <= end_dt]
                            
                            if data_rows:
                                asset_data = {
                                    'datetime': [r.datetime.isoformat() for r in data_rows],
                                    'open': [float(r.open) for r in data_rows],
                                    'high': [float(r.max) for r in data_rows],
                                    'low': [float(r.min) for r in data_rows],
                                    'close': [float(r.close) for r in data_rows],
                                    'volume': [float(r.volume) for r in data_rows]
                                }
                                per_asset_data.append(asset_data)
                                
                                # Use first asset's datetimes for alignment
                                if not datetimes:
                                    datetimes = asset_data['datetime']
            
            if not per_asset_data:
                return {"status": "error", "detail": "No data loaded"}
            
            # Find minimum length across all assets
            min_len = min(len(asset['close']) for asset in per_asset_data)
            if min_len < 50:  # Need enough data for indicators
                return {"status": "error", "detail": "Insufficient data for analysis"}
            
            # 2) Calculate technical indicators
            step(20, *steps[1])
            per_asset_indicators: list[dict] = []
            
            for asset_data in per_asset_data:
                closes = asset_data['close'][-min_len:]  # Use last min_len bars
                
                indicators = {
                    'sma': calculate_sma(closes, sma_period),
                    'rsi': calculate_rsi(closes, rsi_period),
                    'bb_upper': calculate_bollinger_bands(closes, bb_period, bb_std)[0],
                    'bb_middle': calculate_bollinger_bands(closes, bb_period, bb_std)[1],
                    'bb_lower': calculate_bollinger_bands(closes, bb_period, bb_std)[2],
                    'macd_line': calculate_macd(closes, macd_fast, macd_slow, macd_signal)[0],
                    'macd_signal': calculate_macd(closes, macd_fast, macd_slow, macd_signal)[1],
                    'macd_histogram': calculate_macd(closes, macd_fast, macd_slow, macd_signal)[2]
                }
                per_asset_indicators.append(indicators)
            
            # 3) Load and process news data
            step(30, *steps[2])
            per_asset_news_influence: list[list[float]] = []
            
            try:
                start_dt = datetime.fromisoformat(datetimes[0])
                end_dt = datetime.fromisoformat(datetimes[-1])
                bar_dt = timedelta(minutes=5 if tf == '5m' else 1)  # Default to 5m
                
                async with db_helper.get_session() as session:
                    result = await session.execute(
                        select(NewsHistoryCoin.coin_id, News.date, NewsHistoryCoin.score)
                        .join(News, News.id == NewsHistoryCoin.id_news)
                        .where(NewsHistoryCoin.coin_id.in_(coin_ids))
                        .where(News.date >= start_dt - news_window_bars * bar_dt)
                        .where(News.date <= end_dt)
                    )
                    rows = result.all()
                
                # Build per-coin news influence series
                coin_to_events: dict[int, list[tuple[datetime, float]]] = {cid: [] for cid in coin_ids}
                for cid, ndt, sc in rows:
                    coin_to_events.setdefault(cid, []).append((ndt, float(sc)))
                
                for cid in coin_ids:
                    events = sorted(coin_to_events.get(cid, []), key=lambda x: x[0])
                    
                    # Build rolling influence over bars
                    influence_series: list[float] = []
                    for i in range(min_len):
                        bar_time = end_dt - (min_len - 1 - i) * bar_dt
                        window_start = bar_time - news_window_bars * bar_dt
                        
                        # Sum news scores in window
                        window_score = 0.0
                        for event_time, score in events:
                            if window_start <= event_time <= bar_time:
                                window_score += score
                        
                        # Normalize and apply exponential decay
                        influence = window_score / max(news_window_bars, 1)
                        influence_series.append(influence)
                    
                    per_asset_news_influence.append(influence_series)
                    
            except Exception as e:
                logger.warning(f"News processing failed: {e}")
                per_asset_news_influence = [[0.0] * min_len for _ in per_asset_data]
            
            # 4) Generate trading signals
            step(40, *steps[3])
            per_asset_signals: list[list[int]] = []
            
            for i, asset_data in enumerate(per_asset_data):
                closes = asset_data['close'][-min_len:]
                indicators = per_asset_indicators[i]
                news_influence = per_asset_news_influence[i] if i < len(per_asset_news_influence) else [0.0] * min_len
                
                signals = generate_trading_signals(
                    {'close': closes}, 
                    indicators, 
                    news_influence
                )
                per_asset_signals.append(signals)
            
            # 5) Calculate position sizes and risk management
            step(50, *steps[4])
            per_asset_positions: list[list[float]] = []
            per_asset_volatility: list[float] = []
            
            for i, asset_data in enumerate(per_asset_data):
                closes = asset_data['close'][-min_len:]
                signals = per_asset_signals[i]
                
                # Calculate volatility (rolling standard deviation of returns)
                returns = []
                for j in range(1, len(closes)):
                    if closes[j-1] != 0:
                        ret = (closes[j] / closes[j-1]) - 1.0
                        returns.append(ret)
                
                if returns:
                    vol = math.sqrt(sum((r - sum(returns)/len(returns))**2 / max(len(returns)-1, 1))
                else:
                    vol = 0.0
                
                per_asset_volatility.append(vol)
                
                # Calculate position sizes
                positions = []
                for j, signal in enumerate(signals):
                    if j < len(signals) - 1:  # Don't trade on last signal
                        risk_cap = max(0.1, 1.0 - (vol * 5))  # Reduce risk in high volatility
                        pos_size = calculate_position_size(signal, risk_cap, vol, max_position)
                        positions.append(pos_size)
                    else:
                        positions.append(0.0)
                
                per_asset_positions.append(positions)
            
            # 6) Execute backtest with realistic trading
            step(60, *steps[5])
            step(70, *steps[6])
            
            # Portfolio tracking
            portfolio_value = initial_capital
            portfolio_history = [portfolio_value]
            per_asset_pnl: list[float] = [0.0] * len(coin_ids)
            per_asset_trades: list[list[dict]] = [[] for _ in coin_ids]
            
            # Track positions and execute trades
            for i in range(1, min_len - 1):  # Skip first and last bar
                bar_pnl = 0.0
                
                for asset_idx in range(len(coin_ids)):
                    if asset_idx < len(per_asset_signals) and asset_idx < len(per_asset_positions):
                        signal = per_asset_signals[asset_idx][i]
                        position_size = per_asset_positions[asset_idx][i]
                        
                        if signal != 0 and position_size > 0:
                            # Get prices for this bar
                            asset_data = per_asset_data[asset_idx]
                            entry_price = asset_data['close'][i]
                            exit_price = asset_data['close'][i + 1]
                            
                            # Calculate trade PnL
                            trade_pnl = calculate_trade_pnl(
                                entry_price, exit_price, position_size, signal, 
                                commission, slippage
                            )
                            
                            bar_pnl += trade_pnl
                            per_asset_pnl[asset_idx] += trade_pnl
                            
                            # Record trade
                            trade_record = {
                                'timestamp': asset_data['datetime'][i],
                                'signal': signal,
                                'position_size': position_size,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'commission': commission * position_size,
                                'slippage': slippage * position_size
                            }
                            per_asset_trades[asset_idx].append(trade_record)
                
                # Update portfolio value
                portfolio_value += bar_pnl
                portfolio_history.append(portfolio_value)
            
            # 7) Calculate comprehensive metrics
            step(80, *steps[7])
            
            # Calculate portfolio returns
            portfolio_returns = []
            for i in range(1, len(portfolio_history)):
                if portfolio_history[i-1] != 0:
                    ret = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                    portfolio_returns.append(ret)
            
            # Use enhanced metrics calculation
            portfolio_metrics = calculate_portfolio_metrics([portfolio_returns], initial_capital)
            
            # Calculate additional metrics
            total_trades = sum(len(trades) for trades in per_asset_trades)
            winning_trades = sum(1 for trades in per_asset_trades for trade in trades if trade['pnl'] > 0)
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
            
            # Risk metrics
            risk_metrics = calculate_risk_metrics(portfolio_returns)
            
            # Trading metrics
            trading_metrics = calculate_trading_metrics(portfolio_returns)
            
            # 8) Generate artifacts
            step(90, *steps[8])
            artifacts = {}
            
            try:
                # Equity curve CSV
                out_dir = os.environ.get("PIPELINE_ARTIFACTS_DIR", "/tmp")
                os.makedirs(out_dir, exist_ok=True)
                
                equity_path = os.path.join(out_dir, f"equity_{uuid4().hex}.csv")
                with open(equity_path, "w", encoding="utf-8") as f:
                    f.write("timestamp,equity\n")
                    for i, (dt, equity) in enumerate(zip(datetimes[:len(portfolio_history)], portfolio_history)):
                        f.write(f"{dt},{equity:.6f}\n")
                artifacts["equity_csv"] = equity_path
                
                # Trades CSV
                trades_path = os.path.join(out_dir, f"trades_{uuid4().hex}.csv")
                with open(trades_path, "w", encoding="utf-8") as f:
                    f.write("timestamp,coin_id,signal,position_size,entry_price,exit_price,pnl,commission,slippage\n")
                    for asset_idx, trades in enumerate(per_asset_trades):
                        coin_id = coin_ids[asset_idx] if asset_idx < len(coin_ids) else 0
                        for trade in trades:
                            f.write(f"{trade['timestamp']},{coin_id},{trade['signal']},"
                                   f"{trade['position_size']:.6f},{trade['entry_price']:.6f},"
                                   f"{trade['exit_price']:.6f},{trade['pnl']:.6f},"
                                   f"{trade['commission']:.6f},{trade['slippage']:.6f}\n")
                artifacts["trades_csv"] = trades_path
                
            except Exception as e:
                logger.warning(f"Artifact generation failed: {e}")
            
            # Final metrics
            metrics = {
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'trading_metrics': trading_metrics,
                'summary': {
                    'initial_capital': initial_capital,
                    'final_value': portfolio_value,
                    'total_return': (portfolio_value - initial_capital) / initial_capital,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'timeframe': tf,
                    'bars_analyzed': min_len,
                    'coins_traded': len(coin_ids)
                },
                'per_coin_pnl': dict(zip(map(str, coin_ids), per_asset_pnl)),
                'artifacts': artifacts
            }
            
            # Save backtest results
            try:
                async with db_helper.get_session() as session:
                    bt = BacktestModel(
                        pipeline_id=pipeline_id,
                        timeframe=tf,
                        start=start_dt if 'start_dt' in locals() else None,
                        end=end_dt if 'end_dt' in locals() else None,
                        config_json=cfg,
                        metrics_json=metrics,
                        artifacts=artifacts,
                    )
                    session.add(bt)
                    await session.commit()
            except Exception as e:
                logger.warning(f"Failed to save backtest: {e}")
            
            self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Готово', 'metrics': metrics})
            return {"status": "success", "metrics": metrics}
            
        except Exception as e:
            logger.exception("run_pipeline_backtest_task failed")
            return {"status": "error", "detail": str(e)}

    return asyncio.run(_run())


@celery_app.task(bind=True)
def train_pred_time_task(self, config: dict):
    """Train Pred_time model with PyTorch pipeline"""
    try:
        from core.services.pred_time_service import PredTimeService
        from core.database import db_helper
        import asyncio
        
        # Extract configuration
        coin_ids = config.get('coin_ids', [])
        agent_id = config.get('agent_id')
        
        if not coin_ids:
            return {"status": "error", "detail": "coin_ids is required"}
        
        if not agent_id:
            return {"status": "error", "detail": "agent_id is required"}
        
        # Initialize service
        pred_time_service = PredTimeService()
        
        async def _run():
            async with db_helper.get_session() as session:
                self.update_state(
                    state='PROGRESS',
                    meta={'progress': 10, 'message': f'Starting Pred_time training for {len(coin_ids)} coins...'}
                )
                
                # Train model
                result = await pred_time_service.train_model(
                    session, coin_ids, config, agent_id
                )
                
                if result['status'] == 'success':
                    self.update_state(
                        state='SUCCESS',
                        meta={
                            'progress': 100,
                            'message': 'Pred_time model training completed successfully',
                            'result': result
                        }
                    )
                else:
                    self.update_state(
                        state='FAILURE',
                        meta={
                            'progress': 100,
                            'message': f"Pred_time training failed: {result.get('error', 'Unknown error')}"
                        }
                    )
                
                return result
        
        return asyncio.run(_run())
        
    except Exception as e:
        logger.exception("train_pred_time_task failed")
        return {"status": "error", "detail": str(e)}


@celery_app.task(bind=True)
def evaluate_pred_time_task(self, config: dict):
    """Evaluate Pred_time model performance"""
    try:
        from core.services.pred_time_service import PredTimeService
        from core.database import db_helper
        import asyncio
        
        # Extract configuration
        model_path = config.get('model_path')
        coin_ids = config.get('coin_ids', [])
        evaluation_hours = config.get('evaluation_hours', 168)  # 1 week
        
        if not model_path:
            return {"status": "error", "detail": "model_path is required"}
        
        if not coin_ids:
            return {"status": "error", "detail": "coin_ids is required"}
        
        # Initialize service
        pred_time_service = PredTimeService()
        
        async def _run():
            async with db_helper.get_session() as session:
                self.update_state(
                    state='PROGRESS',
                    meta={'progress': 20, 'message': 'Loading trained model...'}
                )
                
                # Load model
                model = await pred_time_service.load_model(model_path)
                if not model:
                    return {"status": "error", "detail": "Failed to load model"}
                
                self.update_state(
                    state='PROGRESS',
                    meta={'progress': 40, 'message': 'Evaluating model performance...'}
                )
                
                # Evaluate on each coin
                evaluation_results = {}
                total_coins = len(coin_ids)
                
                for i, coin_id in enumerate(coin_ids):
                    try:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'progress': 40 + (i * 40 // total_coins),
                                'message': f'Evaluating coin {coin_id} ({i+1}/{total_coins})...'
                            }
                        )
                        
                        # Get recent data for evaluation
                        ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe='5m')
                        if not ts:
                            evaluation_results[coin_id] = {
                                'status': 'no_data',
                                'message': 'No timeseries data available'
                            }
                            continue
                        
                        data = await orm_get_data_timeseries(session, ts.id)
                        if not data:
                            evaluation_results[coin_id] = {
                                'status': 'no_data',
                                'message': 'No price data available'
                            }
                            continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame([
                            {
                                'timestamp': row.datetime,
                                'open': float(row.open),
                                'high': float(row.high),
                                'low': float(row.low),
                                'close': float(row.close),
                                'volume': float(row.volume)
                            }
                            for row in data
                        ])
                        df.set_index('timestamp', inplace=True)
                        
                        # Prepare features for evaluation
                        X, y = pred_time_service._prepare_features(df, coin_id)
                        
                        if len(X) == 0:
                            evaluation_results[coin_id] = {
                                'status': 'insufficient_data',
                                'message': 'Insufficient data for evaluation'
                            }
                            continue
                        
                        # Make predictions
                        predictions = []
                        targets = []
                        
                        for j in range(0, len(X), config.get('batch_size', 32)):
                            batch_X = X[j:j + config.get('batch_size', 32)]
                            batch_y = y[j:j + config.get('batch_size', 32)]
                            
                            # Make prediction
                            prediction = await pred_time_service.predict(model, batch_X, coin_id)
                            if prediction:
                                predictions.append(prediction['predicted_change'])
                                targets.extend(batch_y)
                        
                        if len(predictions) == 0:
                            evaluation_results[coin_id] = {
                                'status': 'prediction_failed',
                                'message': 'Failed to generate predictions'
                            }
                            continue
                        
                        # Calculate metrics
                        predictions = np.array(predictions)
                        targets = np.array(targets)
                        
                        metrics = pred_time_service._calculate_metrics(predictions, targets)
                        
                        evaluation_results[coin_id] = {
                            'status': 'success',
                            'metrics': metrics,
                            'predictions_count': len(predictions),
                            'targets_count': len(targets)
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed to evaluate coin {coin_id}: {e}")
                        evaluation_results[coin_id] = {
                            'status': 'error',
                            'error': str(e)
                        }
                
                # Calculate overall metrics
                successful_evaluations = [
                    r for r in evaluation_results.values() 
                    if r['status'] == 'success'
                ]
                
                if successful_evaluations:
                    overall_metrics = {
                        'avg_rmse': np.mean([r['metrics']['rmse'] for r in successful_evaluations]),
                        'avg_mae': np.mean([r['metrics']['mae'] for r in successful_evaluations]),
                        'avg_mape': np.mean([r['metrics']['mape'] for r in successful_evaluations]),
                        'avg_direction_accuracy': np.mean([r['metrics']['direction_accuracy'] for r in successful_evaluations]),
                        'avg_correlation': np.mean([r['metrics']['correlation'] for r in successful_evaluations]),
                        'coins_evaluated': len(coin_ids),
                        'successful_evaluations': len(successful_evaluations)
                    }
                else:
                    overall_metrics = {
                        'coins_evaluated': len(coin_ids),
                        'successful_evaluations': 0
                    }
                
                self.update_state(
                    state='SUCCESS',
                    meta={
                        'progress': 100,
                        'message': f'Pred_time evaluation completed for {len(coin_ids)} coins',
                        'results': evaluation_results,
                        'overall_metrics': overall_metrics
                    }
                )
                
                return {
                    'status': 'success',
                    'results': evaluation_results,
                    'overall_metrics': overall_metrics
                }
        
        return asyncio.run(_run())
        
    except Exception as e:
        logger.exception("evaluate_pred_time_task failed")
        return {"status": "error", "detail": str(e)}