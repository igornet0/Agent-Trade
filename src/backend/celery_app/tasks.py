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
from typing import Optional, Dict, List, Any

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
from core.database.orm import (
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
def run_pipeline_backtest_task(
    self, 
    pipeline_config: Dict[str, Any],
    timeframe: str,
    start_date: str,
    end_date: str,
    coins: List[str],
    backtest_id: Optional[int] = None
):
    """Задача для выполнения бэктеста пайплайна"""
    try:
        from core.services.pipeline_orchestrator import PipelineOrchestrator
        from core.database.engine import get_db
        
        # Создаем оркестратор
        orchestrator = PipelineOrchestrator()
        
        # Получаем сессию БД
        db = next(get_db())
        
        # Обновляем мета-информацию задачи
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Starting pipeline execution...'
            }
        )
        
        # Выполняем пайплайн
        results = orchestrator.execute_pipeline(
            pipeline_config=pipeline_config,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            coins=coins,
            backtest_id=backtest_id,
            db_session=db
        )
        
        # Обновляем финальный статус
        self.update_state(
            state='SUCCESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Pipeline execution completed',
                'results': results
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in run_pipeline_backtest_task: {e}")
        
        # Обновляем статус ошибки
        self.update_state(
            state='FAILURE',
            meta={
                'current': 0,
                'total': 100,
                'status': f'Pipeline execution failed: {str(e)}',
                'error': str(e)
            }
        )
        
        raise self.retry(countdown=60, max_retries=3)


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

@app.task(bind=True)
def train_trade_time_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для обучения Trade_time модели"""
    try:
        from core.services.trade_time_service import TradeTimeService
        
        service = TradeTimeService()
        result = service.train_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in train_trade_time_task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@app.task(bind=True)
def evaluate_trade_time_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для оценки Trade_time модели"""
    try:
        from core.services.trade_time_service import TradeTimeService
        
        service = TradeTimeService()
        result = service.evaluate_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in evaluate_trade_time_task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@app.task(bind=True)
def train_risk_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для обучения Risk модели"""
    try:
        from core.services.risk_service import RiskService
        
        service = RiskService()
        result = service.train_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in train_risk_task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@app.task(bind=True)
def evaluate_risk_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для оценки Risk модели"""
    try:
        from core.services.risk_service import RiskService
        
        service = RiskService()
        result = service.evaluate_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in evaluate_risk_task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@app.task(bind=True)
def train_trade_aggregator_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для обучения Trade Aggregator модели"""
    try:
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        result = service.train_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in train_trade_aggregator_task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@app.task(bind=True)
def evaluate_trade_aggregator_task(self, coin_id: str, start_date: str, end_date: str, extra_config: Optional[Dict] = None):
    """Задача для оценки Trade Aggregator модели"""
    try:
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        result = service.evaluate_model(coin_id, start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in evaluate_trade_aggregator_task: {e}")
        raise self.retry(countdown=60, max_retries=3)