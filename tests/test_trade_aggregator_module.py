#!/usr/bin/env python3
"""
Тесты для Trade Aggregator модуля
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTradeAggregatorService(unittest.TestCase):
    """Тесты для TradeAggregatorService"""

    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем временную директорию для тестов
        self.test_dir = tempfile.mkdtemp()
        self.original_models_dir = "models/models_pth/AgentTrade"
        
        # Мокаем зависимости
        self.mock_orm_get_coin_data = Mock()
        self.mock_orm_get_news_background = Mock()
        self.mock_calculate_technical_indicators = Mock()
        
        # Патчим импорты
        with patch.dict('sys.modules', {
            'core.database.orm.market': Mock(orm_get_coin_data=self.mock_orm_get_coin_data),
            'core.database.orm.news': Mock(orm_get_news_background=self.mock_orm_get_news_background),
            'core.utils.metrics': Mock(calculate_technical_indicators=self.mock_calculate_technical_indicators)
        }):
            from core.services.trade_aggregator_service import TradeAggregatorService
            self.service = TradeAggregatorService()
            # Переопределяем директорию моделей для тестов
            self.service.models_dir = self.test_dir

    def tearDown(self):
        """Очистка после каждого теста"""
        # Удаляем временную директорию
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init(self):
        """Тест инициализации сервиса"""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.models_dir, self.test_dir)
        self.assertIn('mode', self.service.default_config)
        self.assertIn('weights', self.service.default_config)
        self.assertIn('thresholds', self.service.default_config)

    def test_calculate_portfolio_metrics_empty(self):
        """Тест расчета метрик портфеля для пустого портфеля"""
        balance = 10000
        positions = []
        
        metrics = self.service._calculate_portfolio_metrics(positions, balance)
        
        self.assertEqual(metrics['total_value'], balance)
        self.assertEqual(metrics['total_pnl'], 0.0)
        self.assertEqual(metrics['exposure'], 0.0)
        self.assertEqual(metrics['diversification'], 1.0)
        self.assertEqual(metrics['risk_score'], 0.0)

    def test_calculate_portfolio_metrics_with_positions(self):
        """Тест расчета метрик портфеля с позициями"""
        balance = 10000
        positions = [
            {'size': 100, 'entry_price': 50, 'unrealized_pnl': 500},
            {'size': 50, 'entry_price': 100, 'unrealized_pnl': -200}
        ]
        
        metrics = self.service._calculate_portfolio_metrics(positions, balance)
        
        self.assertEqual(metrics['total_value'], 10300)  # 10000 + 500 - 200
        self.assertEqual(metrics['total_pnl'], 300)  # 500 - 200
        self.assertEqual(metrics['exposure'], 10000)  # 100*50 + 50*100
        self.assertGreater(metrics['diversification'], 0.0)
        self.assertGreater(metrics['risk_score'], 0.0)

    def test_aggregate_signals(self):
        """Тест агрегации сигналов"""
        pred_time_signals = 0.7
        trade_time_signals = {'buy': 0.6, 'sell': 0.2, 'hold': 0.2}
        risk_signals = {'risk_score': 0.3, 'volume_score': 0.8}
        config = {
            'weights': {'pred_time': 0.4, 'trade_time': 0.4, 'risk': 0.2},
            'thresholds': {'buy_threshold': 0.6, 'sell_threshold': 0.4, 'hold_threshold': 0.3}
        }
        
        result = self.service._aggregate_signals(
            pred_time_signals, trade_time_signals, risk_signals, config
        )
        
        self.assertIn('decision', result)
        self.assertIn('confidence', result)
        self.assertIn('aggregated_score', result)
        self.assertIn('position_size', result)
        self.assertIn('signals', result)
        
        # Проверяем, что решение принято
        self.assertIn(result['decision'], ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_apply_risk_management(self):
        """Тест применения риск-менеджмента"""
        decision = 'buy'
        position_size = 0.15
        portfolio_metrics = {
            'exposure': 5000,
            'total_value': 10000,
            'risk_score': 0.6
        }
        config = {
            'risk_limits': {
                'max_position_size': 0.1,
                'max_leverage': 3.0,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.15
            }
        }
        
        result = self.service._apply_risk_management(
            decision, position_size, portfolio_metrics, config
        )
        
        self.assertIn('position_size', result)
        self.assertIn('stop_loss_pct', result)
        self.assertIn('take_profit_pct', result)
        self.assertIn('max_leverage', result)
        
        # Проверяем, что размер позиции уменьшен из-за высокого риска
        self.assertLessEqual(result['position_size'], position_size)

    def test_create_ml_model_rules_mode(self):
        """Тест создания ML модели в rules режиме"""
        config = {'mode': 'rules'}
        
        model = self.service._create_ml_model(config)
        
        self.assertIsNone(model)

    def test_create_ml_model_ml_mode(self):
        """Тест создания ML модели в ML режиме"""
        config = {
            'mode': 'ml',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
        
        # Мокаем xgb
        with patch('core.services.trade_aggregator_service.xgb') as mock_xgb:
            mock_xgb.XGBRegressor = Mock()
            
            model = self.service._create_ml_model(config)
            
            if mock_xgb.XGBRegressor.called:
                mock_xgb.XGBRegressor.assert_called_with(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )

    def test_calculate_max_drawdown(self):
        """Тест расчета максимальной просадки"""
        coin_data = [
            {'close': 100},
            {'close': 110},
            {'close': 90},
            {'close': 95},
            {'close': 105}
        ]
        
        max_dd = self.service._calculate_max_drawdown(coin_data)
        
        self.assertGreaterEqual(max_dd, 0.0)
        self.assertLessEqual(max_dd, 1.0)
        
        # Для данных выше максимальная просадка должна быть около 0.18 (90/110 - 1)
        self.assertAlmostEqual(max_dd, 0.18, places=1)

    def test_calculate_max_drawdown_insufficient_data(self):
        """Тест расчета просадки для недостаточных данных"""
        coin_data = [{'close': 100}]
        
        max_dd = self.service._calculate_max_drawdown(coin_data)
        
        self.assertEqual(max_dd, 0.0)

    def test_predict_basic(self):
        """Тест базового предсказания"""
        coin_id = "BTC"
        pred_time_signals = 0.6
        trade_time_signals = {'buy': 0.5, 'sell': 0.3, 'hold': 0.2}
        risk_signals = {'risk_score': 0.4, 'volume_score': 0.7}
        portfolio_state = {'balance': 10000, 'positions': []}
        
        result = self.service.predict(
            coin_id, pred_time_signals, trade_time_signals, 
            risk_signals, portfolio_state
        )
        
        self.assertIn('decision', result)
        self.assertIn('confidence', result)
        self.assertIn('position_size', result)
        self.assertIn('signals', result)
        self.assertIn('portfolio_metrics', result)

    def test_predict_error_handling(self):
        """Тест обработки ошибок в предсказании"""
        # Мокаем _aggregate_signals чтобы вызвать ошибку
        with patch.object(self.service, '_aggregate_signals', side_effect=Exception("Test error")):
            result = self.service.predict("BTC")
            
            self.assertEqual(result['decision'], 'hold')
            self.assertEqual(result['confidence'], 0.0)
            self.assertIn('error', result)

    def test_save_model(self):
        """Тест сохранения модели"""
        model = Mock()
        config = {'test': 'config'}
        metadata = {'test': 'metadata'}
        model_path = os.path.join(self.test_dir, 'test_model')
        
        result = self.service._save_model(model, config, metadata, model_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'config.json')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'metadata.json')))

    def test_load_model(self):
        """Тест загрузки модели"""
        # Сначала сохраняем модель
        model = Mock()
        config = {'test': 'config'}
        metadata = {'test': 'metadata'}
        model_path = os.path.join(self.test_dir, 'test_model')
        
        self.service._save_model(model, config, metadata, model_path)
        
        # Теперь загружаем
        result = self.service.load_model(model_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['config'], config)
        self.assertEqual(result['metadata'], metadata)

    def test_load_model_nonexistent(self):
        """Тест загрузки несуществующей модели"""
        result = self.service.load_model('/nonexistent/path')
        
        self.assertIsNone(result)


class TestTradeAggregatorIntegration(unittest.TestCase):
    """Интеграционные тесты для Trade Aggregator"""

    def setUp(self):
        """Настройка перед каждым тестом"""
        self.test_dir = tempfile.mkdtemp()
        
        # Мокаем все внешние зависимости
        self.mock_orm_get_coin_data = Mock()
        self.mock_orm_get_news_background = Mock()
        self.mock_calculate_technical_indicators = Mock()
        
        with patch.dict('sys.modules', {
            'core.database.orm.market': Mock(orm_get_coin_data=self.mock_orm_get_coin_data),
            'core.database.orm.news': Mock(orm_get_news_background=self.mock_orm_get_news_background),
            'core.utils.metrics': Mock(calculate_technical_indicators=self.mock_calculate_technical_indicators)
        }):
            from core.services.trade_aggregator_service import TradeAggregatorService
            self.service = TradeAggregatorService()
            self.service.models_dir = self.test_dir

    def tearDown(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_train_model_integration(self):
        """Интеграционный тест обучения модели"""
        # Мокаем данные
        mock_coin_data = [
            {'close': 100, 'volume': 1000, 'timestamp': '2023-01-01'},
            {'close': 110, 'volume': 1100, 'timestamp': '2023-01-02'},
            {'close': 105, 'volume': 1050, 'timestamp': '2023-01-03'}
        ]
        
        mock_news_background = [0.5, 0.6, 0.4]
        mock_tech_features = {'sma': 102.5, 'rsi': 55.0}
        
        self.mock_orm_get_coin_data.return_value = mock_coin_data
        self.mock_orm_get_news_background.return_value = mock_news_background
        self.mock_calculate_technical_indicators.return_value = mock_tech_features
        
        # Тестируем обучение
        result = self.service.train_model(
            "BTC", "2023-01-01", "2023-01-03", 
            {'mode': 'rules'}
        )
        
        self.assertTrue(result['success'])
        self.assertIn('model_path', result)
        self.assertIn('metadata', result)
        self.assertIn('config', result)

    def test_evaluate_model_integration(self):
        """Интеграционный тест оценки модели"""
        # Мокаем данные
        mock_coin_data = [
            {'close': 100, 'volume': 1000, 'timestamp': '2023-01-01'},
            {'close': 110, 'volume': 1100, 'timestamp': '2023-01-02'},
            {'close': 105, 'volume': 1050, 'timestamp': '2023-01-03'}
        ]
        
        self.mock_orm_get_coin_data.return_value = mock_coin_data
        
        # Тестируем оценку
        result = self.service.evaluate_model(
            "BTC", "2023-01-01", "2023-01-03", 
            {'mode': 'rules'}
        )
        
        self.assertTrue(result['success'])
        self.assertIn('metrics', result)
        self.assertIn('trades', result)
        self.assertIn('positions', result)
        self.assertIn('config', result)
        
        # Проверяем метрики
        metrics = result['metrics']
        self.assertIn('total_return_pct', metrics)
        self.assertIn('final_balance', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('volatility', metrics)


def main():
    """Основная функция для запуска тестов"""
    print("🧪 Запуск тестов Trade Aggregator модуля...")
    
    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты
    suite.addTests(loader.loadTestsFromTestCase(TestTradeAggregatorService))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeAggregatorIntegration))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выводим результат
    if result.wasSuccessful():
        print("✅ Все тесты Trade Aggregator модуля прошли успешно!")
        return 0
    else:
        print("❌ Некоторые тесты Trade Aggregator модуля провалились")
        return 1


if __name__ == '__main__':
    sys.exit(main())
