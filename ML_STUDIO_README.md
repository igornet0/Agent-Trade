# ML Studio - Интерфейс для обучения и тестирования ML-моделей

## Обзор

ML Studio предоставляет комплексный интерфейс для управления машинным обучением в торговой системе. Интерфейс позволяет администраторам обучать, тестировать и управлять различными типами ML-агентов.

## Архитектура ML-системы

### 1. News Model (Агент новостей)
- **Вход**: Текст новости, источник, время публикации
- **Выход**: 
  - Список монет, на которые влияет новость
  - Коэффициент влияния от -100 до +100
- **Хранение**: Обновляет новостной фон для каждой монеты
- **Использование**: В Pred_time, Trade_time, Risk, Trade

### 2. Pred_time Model (Прогноз цены)
- **Вход**:
  - Временной ряд по монете (OHLCV, 5m интервал)
  - Технические индикаторы (SMA, RSI, MACD, Bollinger)
  - Новостной фон (из News Model)
- **Выход**:
  - Предсказание изменения цены на N шагов вперёд
  - Вероятность сценариев (рост/падение/боковик)
- **Модели**: LSTM/GRU, Transformer Time Series, ансамбли

### 3. Trade_time Model (Торговые сигналы)
- **Вход**:
  - Временной ряд OHLCV
  - Прогноз цены от Pred_time
  - Новостной фон
- **Выход**: Вероятности {hold, buy, sell}
- **Модели**: Классификаторы (LightGBM, CatBoost, Transformer, RL)

### 4. Risk Model (Оценка риска)
- **Вход**:
  - Открытые позиции (PnL, маржа, плечо)
  - Доступный баланс
  - Предсказания от ансамбля Trade_time моделей
  - Новостной фон
- **Выход**:
  - Риск сделки в %
  - Максимальный допустимый объем для заявки
- **Модели**: Эвристическая + ML (XGBoost)

### 5. Trade Model (Исполнитель)
- **Вход**: Сигналы от всех моделей
- **Выход**:
  - Итоговое решение (buy/sell/hold)
  - Объем сделки
  - Обновление портфеля
- **Особенности**: RL-агент с reward = доходность портфеля

## Компоненты интерфейса

### 1. Модели (🤖)
- Просмотр всех обученных моделей
- Фильтрация по типу агента
- Мониторинг статуса обучения
- Возможность оценки и продвижения моделей

### 2. Обучение (🎓)
- Создание новых моделей с детальными настройками
- Выбор типа агента и конфигурации
- Настройка гиперпараметров обучения
- Выбор монет для обучения
- Мониторинг прогресса обучения

### 3. Тестирование (🧪)
- Выбор модели для тестирования
- Настройка параметров тестирования
- Выбор метрик для оценки
- Визуализация результатов
- Рекомендации по улучшению

### 4. Данные (📊)
- Статистика доступных данных
- Экспорт/импорт данных
- Анализ качества данных
- Выбор временных диапазонов

### 5. Новости (📰)
- Управление новостным фоном
- Пересчет влияния новостей
- Просмотр новостного фона по монетам

## Использование

### Обучение новой модели

1. Перейдите на вкладку "Обучение"
2. Выберите тип агента:
   - **News Model**: Для анализа новостей
   - **Pred_time Model**: Для прогнозирования цены
   - **Trade_time Model**: Для генерации торговых сигналов
   - **Risk Model**: Для оценки рисков
   - **Trade Aggregator**: Для финального агрегатора

3. Настройте параметры:
   - **Общие**: название, таймфрейм, эпохи, batch size, learning rate
   - **Специфичные**: зависят от типа агента

4. Выберите монеты для обучения
5. Запустите обучение

### Тестирование модели

1. Перейдите на вкладку "Тестирование"
2. Выберите модель для тестирования
3. Настройте параметры тестирования:
   - Временной диапазон
   - Монеты для тестирования
   - Метрики для оценки
4. Запустите тестирование
5. Просмотрите результаты

### Управление данными

1. Перейдите на вкладку "Данные"
2. Выберите временной диапазон и монеты
3. Получите статистику данных
4. Экспортируйте данные при необходимости
5. Импортируйте новые данные

## API Endpoints

### Модели
- `GET /api_db_agent/models` - Список моделей
- `POST /api_db_agent/{type}/train` - Обучение модели
- `POST /api_db_agent/test_model` - Тестирование модели
- `GET /api_db_agent/task_status/{taskId}` - Статус задачи

### Данные
- `GET /api_db_agent/data/stats` - Статистика данных
- `GET /api_db_agent/data/export` - Экспорт данных
- `POST /api_db_agent/data/import` - Импорт данных

### Новости
- `POST /api_db_agent/news/recalc_background` - Пересчет новостного фона
- `GET /api_db_agent/news/background/{coinId}` - Новостной фон монеты

## Структура данных

### DataTimeseries
```sql
CREATE TABLE data_timeseries (
    id INTEGER PRIMARY KEY,
    timeseries_id INTEGER,
    datetime TIMESTAMP,
    open FLOAT,
    max FLOAT,
    min FLOAT,
    close FLOAT,
    volume FLOAT
);
```

### Agent
```sql
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50),
    type VARCHAR(50),
    timeframe VARCHAR(50),
    status VARCHAR(20),
    version VARCHAR(20)
);
```

### AgentTrain
```sql
CREATE TABLE agent_trains (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    task_id VARCHAR,
    epochs INTEGER,
    epoch_now INTEGER,
    loss_now FLOAT,
    status VARCHAR(20)
);
```

## Конфигурации по умолчанию

### News Model
```json
{
  "model_name": "finbert",
  "window_hours": 24,
  "decay_factor": 0.95,
  "min_confidence": 0.7,
  "use_sentiment": true,
  "use_entities": true
}
```

### Pred_time Model
```json
{
  "model_name": "LSTM",
  "seq_len": 96,
  "pred_len": 12,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "use_news_background": true,
  "indicators": ["SMA", "RSI", "MACD", "Bollinger"],
  "target": "price_change"
}
```

### Trade_time Model
```json
{
  "classifier": "LightGBM",
  "target_scheme": "direction3",
  "use_news_background": true,
  "use_pred_time": true,
  "features": ["price", "volume", "indicators", "news_bg"],
  "threshold": 0.6
}
```

### Risk Model
```json
{
  "model_name": "XGBoost",
  "max_leverage": 3,
  "risk_limit_pct": 2,
  "max_drawdown_pct": 20,
  "per_trade_risk_pct": 1,
  "features": ["balance", "pnl", "leverage", "signals", "market_volatility"]
}
```

### Trade Aggregator
```json
{
  "mode": "rules",
  "weights": {
    "pred_time": 0.4,
    "trade_time": 0.4,
    "news": 0.1,
    "risk": 0.1
  },
  "rl_enabled": false,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "min_confidence": 0.7
}
```

## Метрики оценки

### News Model
- accuracy, precision, recall, f1_score, sentiment_accuracy

### Pred_time Model
- mae, mse, rmse, mape, directional_accuracy

### Trade_time Model
- accuracy, precision, recall, f1_score, profit_factor

### Risk Model
- risk_accuracy, max_drawdown, sharpe_ratio, calmar_ratio

### Trade Aggregator
- total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor

## Разработка

### Добавление нового типа агента

1. Добавьте тип в `AGENT_TYPES` в `ModelTrainForm.jsx`
2. Создайте конфигурацию по умолчанию в `DEFAULT_CONFIGS`
3. Добавьте специфичные настройки в `renderConfigSection()`
4. Добавьте метрики в `TEST_METRICS` в `ModelTester.jsx`
5. Обновите API endpoints

### Добавление новых метрик

1. Добавьте метрику в соответствующий массив `TEST_METRICS`
2. Обновите функцию `formatMetric()` для правильного отображения
3. Обновите функцию `getMetricColor()` для цветового кодирования

### Кастомизация визуализации

1. Создайте новый компонент в `components/ml/`
2. Добавьте необходимые API endpoints
3. Интегрируйте компонент в соответствующий таб

## Мониторинг и логирование

- Все задачи обучения и тестирования выполняются асинхронно
- Прогресс отслеживается через polling API
- Результаты сохраняются в базе данных
- Логи доступны через API endpoints

## Безопасность

- Доступ только для администраторов
- Валидация входных данных
- Ограничения на размер загружаемых файлов
- Логирование всех операций

## Производительность

- Асинхронная обработка тяжелых задач
- Кэширование результатов
- Пагинация для больших списков
- Оптимизированные запросы к базе данных
