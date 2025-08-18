"""
Pipeline Orchestrator - реальное выполнение ML пайплайнов
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from ..database.orm.market import orm_get_coin_data
from ..database.orm.news import orm_get_news_background
from ..database.orm.pipelines import orm_create_backtest, orm_update_backtest_status
from ..utils.metrics import calculate_portfolio_metrics, calculate_regression_metrics
from ..utils.metrics import calculate_classification_metrics, calculate_risk_metrics, calculate_trading_metrics

# Импорты ML сервисов
from .news_background_service import NewsBackgroundService
from .pred_time_service import PredTimeService
from .trade_time_service import TradeTimeService
from .risk_service import RiskService
from .trade_aggregator_service import TradeAggregatorService

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Оркестратор для выполнения ML пайплайнов"""
    
    def __init__(self):
        self.news_service = NewsBackgroundService()
        self.pred_time_service = PredTimeService()
        self.trade_time_service = TradeTimeService()
        self.risk_service = RiskService()
        self.trade_aggregator_service = TradeAggregatorService()
        
        # Создаем директории для артефактов
        self.artifacts_dir = "artifacts/pipelines"
        os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def execute_pipeline(
        self,
        pipeline_config: Dict[str, Any],
        timeframe: str,
        start_date: str,
        end_date: str,
        coins: List[str],
        backtest_id: Optional[int] = None,
        db_session=None
    ) -> Dict[str, Any]:
        """
        Выполнение полного пайплайна ML системы
        
        Args:
            pipeline_config: Конфигурация пайплайна (узлы и связи)
            timeframe: Таймфрейм данных
            start_date: Начальная дата
            end_date: Конечная дата
            coins: Список монет для обработки
            backtest_id: ID бэктеста для обновления прогресса
            db_session: Сессия БД
        
        Returns:
            Результаты выполнения пайплайна
        """
        try:
            logger.info(f"Starting pipeline execution for {len(coins)} coins")
            
            # Инициализация результатов
            results = {
                "pipeline_id": pipeline_config.get("id"),
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "coins": coins,
                "execution_time": datetime.utcnow().isoformat(),
                "nodes": {},
                "metrics": {},
                "artifacts": {}
            }
            
            # Обновляем прогресс
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.1)
            
            # Этап 1: Загрузка данных
            logger.info("Stage 1: Loading market data")
            market_data = self._load_market_data(coins, timeframe, start_date, end_date)
            results["nodes"]["data_loading"] = {
                "status": "completed",
                "data_points": sum(len(data) for data in market_data.values())
            }
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.2)
            
            # Этап 2: Новостной фон
            logger.info("Stage 2: Processing news background")
            news_background = self._process_news_background(coins, start_date, end_date)
            results["nodes"]["news_background"] = {
                "status": "completed",
                "background_points": len(news_background)
            }
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.3)
            
            # Этап 3: Pred_time модели
            logger.info("Stage 3: Running Pred_time models")
            pred_results = self._run_pred_time_models(market_data, news_background, pipeline_config)
            results["nodes"]["pred_time"] = pred_results
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.5)
            
            # Этап 4: Trade_time модели
            logger.info("Stage 4: Running Trade_time models")
            trade_results = self._run_trade_time_models(market_data, pred_results, news_background, pipeline_config)
            results["nodes"]["trade_time"] = trade_results
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.7)
            
            # Этап 5: Risk модели
            logger.info("Stage 5: Running Risk models")
            risk_results = self._run_risk_models(market_data, trade_results, news_background, pipeline_config)
            results["nodes"]["risk"] = risk_results
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.8)
            
            # Этап 6: Trade Aggregator
            logger.info("Stage 6: Running Trade Aggregator")
            trade_agg_results = self._run_trade_aggregator(
                market_data, pred_results, trade_results, risk_results, pipeline_config
            )
            results["nodes"]["trade_aggregator"] = trade_agg_results
            
            if backtest_id and db_session:
                orm_update_backtest_status(db_session, backtest_id, "running", 0.9)
            
            # Этап 7: Расчет финальных метрик
            logger.info("Stage 7: Calculating final metrics")
            final_metrics = self._calculate_final_metrics(
                market_data, pred_results, trade_results, risk_results, trade_agg_results
            )
            results["metrics"] = final_metrics
            
            # Сохранение артефактов
            artifacts = self._save_artifacts(results, pipeline_config.get("id"))
            results["artifacts"] = artifacts
            
            logger.info("Pipeline execution completed successfully")
            
            # Обновляем статус бэктеста
            if backtest_id and db_session:
                orm_update_backtest_status(
                    db_session, backtest_id, "completed", 1.0,
                    metrics_json=final_metrics, artifacts=artifacts
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            
            # Обновляем статус ошибки
            if backtest_id and db_session:
                orm_update_backtest_status(
                    db_session, backtest_id, "failed", 0.0,
                    error_message=str(e)
                )
            
            raise
    
    def _load_market_data(
        self, 
        coins: List[str], 
        timeframe: str, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Загрузка рыночных данных для всех монет"""
        market_data = {}
        
        for coin in coins:
            try:
                # Используем существующий ORM метод
                data = orm_get_coin_data(coin, timeframe, start_date, end_date)
                if data is not None and not data.empty:
                    market_data[coin] = data
                    logger.info(f"Loaded {len(data)} data points for {coin}")
                else:
                    logger.warning(f"No data found for {coin}")
            except Exception as e:
                logger.error(f"Error loading data for {coin}: {e}")
        
        return market_data
    
    def _process_news_background(
        self, 
        coins: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Обработка новостного фона"""
        news_background = {}
        
        for coin in coins:
            try:
                # Используем существующий ORM метод
                background = orm_get_news_background(coin, start_date, end_date)
                if background is not None and not background.empty:
                    news_background[coin] = background
                    logger.info(f"Loaded news background for {coin}")
                else:
                    logger.warning(f"No news background found for {coin}")
            except Exception as e:
                logger.error(f"Error loading news background for {coin}: {e}")
        
        return news_background
    
    def _run_pred_time_models(
        self,
        market_data: Dict[str, pd.DataFrame],
        news_background: Dict[str, pd.DataFrame],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение Pred_time моделей"""
        pred_results = {}
        
        for coin, data in market_data.items():
            try:
                # Получаем конфигурацию для Pred_time
                pred_config = pipeline_config.get("pred_time_config", {})
                
                # Объединяем данные с новостным фоном
                if coin in news_background:
                    merged_data = self._merge_market_news_data(data, news_background[coin])
                else:
                    merged_data = data
                
                # Выполняем предсказание
                prediction = self.pred_time_service.predict(merged_data, pred_config)
                
                pred_results[coin] = {
                    "predictions": prediction.get("predictions", []),
                    "confidence": prediction.get("confidence", 0.0),
                    "metrics": prediction.get("metrics", {})
                }
                
                logger.info(f"Pred_time completed for {coin}")
                
            except Exception as e:
                logger.error(f"Error in Pred_time for {coin}: {e}")
                pred_results[coin] = {"error": str(e)}
        
        return pred_results
    
    def _run_trade_time_models(
        self,
        market_data: Dict[str, pd.DataFrame],
        pred_results: Dict[str, Any],
        news_background: Dict[str, pd.DataFrame],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение Trade_time моделей"""
        trade_results = {}
        
        for coin, data in market_data.items():
            try:
                # Получаем конфигурацию для Trade_time
                trade_config = pipeline_config.get("trade_time_config", {})
                
                # Объединяем данные с предсказаниями и новостным фоном
                features = self._prepare_trade_features(data, pred_results.get(coin, {}), news_background.get(coin))
                
                # Выполняем классификацию
                classification = self.trade_time_service.predict(features, trade_config)
                
                trade_results[coin] = {
                    "signals": classification.get("signals", []),
                    "probabilities": classification.get("probabilities", []),
                    "metrics": classification.get("metrics", {})
                }
                
                logger.info(f"Trade_time completed for {coin}")
                
            except Exception as e:
                logger.error(f"Error in Trade_time for {coin}: {e}")
                trade_results[coin] = {"error": str(e)}
        
        return trade_results
    
    def _run_risk_models(
        self,
        market_data: Dict[str, pd.DataFrame],
        trade_results: Dict[str, Any],
        news_background: Dict[str, pd.DataFrame],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение Risk моделей"""
        risk_results = {}
        
        for coin, data in market_data.items():
            try:
                # Получаем конфигурацию для Risk
                risk_config = pipeline_config.get("risk_config", {})
                
                # Подготавливаем данные для риск-модели
                risk_features = self._prepare_risk_features(data, trade_results.get(coin, {}), news_background.get(coin))
                
                # Выполняем оценку риска
                risk_assessment = self.risk_service.predict(risk_features, risk_config)
                
                risk_results[coin] = {
                    "risk_score": risk_assessment.get("risk_score", 0.0),
                    "position_size": risk_assessment.get("position_size", 0.0),
                    "stop_loss": risk_assessment.get("stop_loss", 0.0),
                    "take_profit": risk_assessment.get("take_profit", 0.0)
                }
                
                logger.info(f"Risk assessment completed for {coin}")
                
            except Exception as e:
                logger.error(f"Error in Risk assessment for {coin}: {e}")
                risk_results[coin] = {"error": str(e)}
        
        return risk_results
    
    def _run_trade_aggregator(
        self,
        market_data: Dict[str, pd.DataFrame],
        pred_results: Dict[str, Any],
        trade_results: Dict[str, Any],
        risk_results: Dict[str, Any],
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение Trade Aggregator"""
        try:
            # Получаем конфигурацию для Trade Aggregator
            aggregator_config = pipeline_config.get("trade_aggregator_config", {})
            
            # Подготавливаем все сигналы
            all_signals = {
                "pred_time": pred_results,
                "trade_time": trade_results,
                "risk": risk_results
            }
            
            # Выполняем агрегацию и финальное решение
            final_decision = self.trade_aggregator_service.predict(all_signals, aggregator_config)
            
            logger.info("Trade Aggregator completed")
            
            return {
                "final_decision": final_decision.get("decision", "hold"),
                "position_size": final_decision.get("position_size", 0.0),
                "confidence": final_decision.get("confidence", 0.0),
                "portfolio_metrics": final_decision.get("portfolio_metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Error in Trade Aggregator: {e}")
            return {"error": str(e)}
    
    def _calculate_final_metrics(
        self,
        market_data: Dict[str, pd.DataFrame],
        pred_results: Dict[str, Any],
        trade_results: Dict[str, Any],
        risk_results: Dict[str, Any],
        trade_agg_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Расчет финальных метрик"""
        metrics = {
            "overall": {},
            "per_coin": {},
            "portfolio": {}
        }
        
        # Агрегируем метрики по монетам
        for coin in market_data.keys():
            coin_metrics = {}
            
            # Метрики Pred_time
            if coin in pred_results and "metrics" in pred_results[coin]:
                coin_metrics["pred_time"] = pred_results[coin]["metrics"]
            
            # Метрики Trade_time
            if coin in trade_results and "metrics" in trade_results[coin]:
                coin_metrics["trade_time"] = trade_results[coin]["metrics"]
            
            # Метрики Risk
            if coin in risk_results:
                coin_metrics["risk"] = {
                    "risk_score": risk_results[coin].get("risk_score", 0.0),
                    "position_size": risk_results[coin].get("position_size", 0.0)
                }
            
            metrics["per_coin"][coin] = coin_metrics
        
        # Портфельные метрики
        if "portfolio_metrics" in trade_agg_results:
            metrics["portfolio"] = trade_agg_results["portfolio_metrics"]
        
        # Общие метрики
        metrics["overall"] = {
            "total_coins": len(market_data),
            "successful_predictions": sum(1 for r in pred_results.values() if "error" not in r),
            "successful_trades": sum(1 for r in trade_results.values() if "error" not in r),
            "average_risk_score": np.mean([r.get("risk_score", 0.0) for r in risk_results.values() if "error" not in r])
        }
        
        return metrics
    
    def _save_artifacts(self, results: Dict[str, Any], pipeline_id: Optional[int]) -> Dict[str, str]:
        """Сохранение артефактов пайплайна"""
        artifacts = {}
        
        try:
            # Создаем директорию для артефактов
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            pipeline_dir = f"{self.artifacts_dir}/pipeline_{pipeline_id or 'manual'}_{timestamp}"
            os.makedirs(pipeline_dir, exist_ok=True)
            
            # Сохраняем результаты в JSON
            results_file = f"{pipeline_dir}/results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            artifacts["results"] = results_file
            
            # Сохраняем метрики отдельно
            metrics_file = f"{pipeline_dir}/metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(results.get("metrics", {}), f, indent=2, default=str)
            artifacts["metrics"] = metrics_file
            
            # Сохраняем конфигурацию узлов
            nodes_file = f"{pipeline_dir}/nodes.json"
            with open(nodes_file, 'w') as f:
                json.dump(results.get("nodes", {}), f, indent=2, default=str)
            artifacts["nodes"] = nodes_file
            
            logger.info(f"Artifacts saved to {pipeline_dir}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
        
        return artifacts
    
    def _merge_market_news_data(
        self, 
        market_data: pd.DataFrame, 
        news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Объединение рыночных данных с новостным фоном"""
        try:
            # Простое объединение по времени (можно улучшить)
            merged = market_data.copy()
            if not news_data.empty:
                # Добавляем колонку новостного фона
                merged['news_background'] = news_data.get('score', 0.0)
            return merged
        except Exception as e:
            logger.error(f"Error merging market and news data: {e}")
            return market_data
    
    def _prepare_trade_features(
        self,
        market_data: pd.DataFrame,
        pred_results: Dict[str, Any],
        news_background: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Подготовка признаков для Trade_time модели"""
        features = market_data.copy()
        
        # Добавляем предсказания
        if pred_results and "predictions" in pred_results:
            features['prediction'] = pred_results["predictions"]
            features['prediction_confidence'] = pred_results.get("confidence", 0.0)
        
        # Добавляем новостной фон
        if news_background is not None and not news_background.empty:
            features['news_background'] = news_background.get('score', 0.0)
        
        return features
    
    def _prepare_risk_features(
        self,
        market_data: pd.DataFrame,
        trade_results: Dict[str, Any],
        news_background: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Подготовка признаков для Risk модели"""
        features = market_data.copy()
        
        # Добавляем торговые сигналы
        if trade_results and "signals" in trade_results:
            features['trade_signal'] = trade_results["signals"]
            features['signal_probability'] = trade_results.get("probabilities", [0.0])
        
        # Добавляем новостной фон
        if news_background is not None and not news_background.empty:
            features['news_background'] = news_background.get('score', 0.0)
        
        return features
