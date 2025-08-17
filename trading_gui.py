# GUI интерфейс для обучения торговых агентов и тестирования в Sandbox
# Использует Streamlit для создания веб-интерфейса

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

# Импортируем нашу систему обучения
from train_agents_complete import (
    AdaptiveLearningAgent, 
    create_and_train_agent, 
    test_agent_trading,
    TradingLabelGenerator
)

# Настройка страницы
st.set_page_config(
    page_title="Trading Agent Training System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🚀 Система обучения торговых агентов для криптовалюты")
st.markdown("---")

# Инициализация сессии
if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'training_status' not in st.session_state:
    st.session_state.training_status = {}
if 'sandbox_mode' not in st.session_state:
    st.session_state.sandbox_mode = False
if 'sandbox_data' not in st.session_state:
    st.session_state.sandbox_data = {}

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор монеты
    coin_options = ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "SOL", "MATIC"]
    selected_coin = st.selectbox("Выберите монету:", coin_options)
    
    # Параметры модели
    st.subheader("Параметры модели")
    seq_len = st.slider("Длина последовательности:", 20, 100, 50)
    pred_len = st.slider("Количество предсказаний:", 3, 10, 5)
    learning_rate = st.selectbox("Learning Rate:", [0.001, 0.0005, 0.0001], index=0)
    batch_size = st.selectbox("Batch Size:", [16, 32, 64], index=1)
    epochs = st.slider("Количество эпох:", 5, 50, 10)
    
    # Параметры торговли
    st.subheader("Параметры торговли")
    profit_threshold = st.slider("Порог прибыли (%):", 1.0, 5.0, 2.0, 0.1)
    stop_loss = st.slider("Стоп-лосс (%):", 0.5, 3.0, 1.0, 0.1)
    take_profit = st.slider("Тейк-профит (%):", 2.0, 8.0, 4.0, 0.1)
    
    # Кнопки управления
    st.markdown("---")
    if st.button("🚀 Обучить агента", type="primary"):
        st.session_state.train_agent = True
    
    if st.button("🧪 Запустить Sandbox"):
        st.session_state.sandbox_mode = True
    
    if st.button("📊 Показать результаты"):
        st.session_state.show_results = True

# Основной контент
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Обучение торгового агента")
    
    # Информация о выбранной монете
    st.subheader(f"Монета: {selected_coin}")
    
    # Создаем демо-данные (в реальном использовании загружайте из вашего источника)
    if selected_coin not in st.session_state.sandbox_data:
        # Генерируем демо-данные для примера
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='5T')
        np.random.seed(42)
        
        # Базовые цены
        base_price = 100 if selected_coin == "BTC" else 50
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        demo_data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.randn(len(dates)) * 0.002),
            'close': prices,
            'max': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.005)),
            'min': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.005)),
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        st.session_state.sandbox_data[selected_coin] = demo_data
    
    # Показываем данные
    data = st.session_state.sandbox_data[selected_coin]
    st.write(f"Данные: {len(data)} записей с {data['datetime'].min()} по {data['datetime'].max()}")
    
    # График цен
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data['datetime'],
        open=data['open'],
        high=data['max'],
        low=data['min'],
        close=data['close'],
        name='Цены'
    ))
    fig.update_layout(
        title=f"График цен {selected_coin}",
        xaxis_title="Время",
        yaxis_title="Цена",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("📊 Статистика")
    
    if selected_coin in st.session_state.agents:
        agent = st.session_state.agents[selected_coin]
        performance = agent.get_performance_summary()
        
        st.metric("Всего сделок", performance.get('total_trades', 0))
        st.metric("Успешных сделок", performance.get('successful_trades', 0))
        st.metric("Процент успеха", f"{performance.get('success_rate', 0):.1%}")
        st.metric("Общий P&L", f"{performance.get('total_pnl', 0):.4f}")
        st.metric("Средний P&L", f"{performance.get('average_pnl', 0):.4f}")
        
        # Веса моделей
        st.subheader("⚖️ Веса моделей")
        weights = performance.get('model_weights', [])
        for i, weight in enumerate(weights):
            st.progress(weight)
            st.caption(f"Модель {i+1}: {weight:.3f}")
    else:
        st.info("Агент еще не обучен")

# Обучение агента
if st.session_state.get('train_agent', False):
    st.markdown("---")
    st.header("🎯 Обучение агента")
    
    with st.spinner("Обучаем агента..."):
        try:
            # Создаем и обучаем агента
            agent = create_and_train_agent(selected_coin, data, device='cpu')
            
            if agent:
                # Настраиваем параметры
                agent.seq_len = seq_len
                agent.pred_len = pred_len
                agent.learning_rate = learning_rate
                agent.batch_size = batch_size
                agent.epochs = epochs
                agent.stop_loss = stop_loss / 100
                agent.take_profit = take_profit / 100
                
                # Сохраняем агента
                st.session_state.agents[selected_coin] = agent
                st.session_state.training_status[selected_coin] = "completed"
                
                st.success(f"✅ Агент {selected_coin} успешно обучен!")
                
                # Показываем информацию о моделях
                st.subheader("📊 Информация о моделях")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Количество моделей", len(agent.ensemble.models))
                    st.metric("Размер входных признаков", agent.input_size)
                
                with col2:
                    st.metric("Длина последовательности", agent.seq_len)
                    st.metric("Количество предсказаний", agent.pred_len)
                
                with col3:
                    st.metric("Learning Rate", agent.learning_rate)
                    st.metric("Batch Size", agent.batch_size)
                
            else:
                st.error("❌ Не удалось создать агента")
                
        except Exception as e:
            st.error(f"❌ Ошибка при обучении: {e}")
    
    st.session_state.train_agent = False

# Sandbox режим
if st.session_state.sandbox_mode:
    st.markdown("---")
    st.header("🧪 Sandbox - Тестирование агента")
    
    if selected_coin in st.session_state.agents:
        agent = st.session_state.agents[selected_coin]
        
        # Параметры тестирования
        col1, col2 = st.columns(2)
        with col1:
            test_days = st.slider("Количество дней для тестирования:", 1, 30, 7)
            start_test = st.button("🚀 Начать тестирование")
        
        with col2:
            if st.button("⏹️ Остановить тестирование"):
                st.session_state.sandbox_mode = False
                st.rerun()
        
        if start_test:
            with st.spinner(f"Тестируем агента на {test_days} днях..."):
                try:
                    # Тестируем агента
                    test_results = test_agent_trading(agent, test_period_days=test_days)
                    
                    if test_results:
                        st.success(f"✅ Тестирование завершено! Выполнено {len(test_results)} сделок")
                        
                        # Показываем результаты
                        st.subheader("📊 Результаты тестирования")
                        
                        # Создаем DataFrame с результатами
                        results_df = pd.DataFrame(test_results)
                        
                        # График P&L
                        if 'profit_loss' in results_df.columns:
                            fig = px.line(
                                results_df, 
                                x='timestamp', 
                                y='profit_loss',
                                title="P&L по сделкам"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Таблица сделок
                        st.subheader("📋 Детали сделок")
                        st.dataframe(results_df)
                        
                        # Обновляем производительность
                        performance = agent.get_performance_summary()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Всего сделок", performance['total_trades'])
                            st.metric("Успешных сделок", performance['successful_trades'])
                        
                        with col2:
                            st.metric("Процент успеха", f"{performance['success_rate']:.1%}")
                            st.metric("Общий P&L", f"{performance['total_pnl']:.4f}")
                        
                        with col3:
                            st.metric("Средний P&L", f"{performance['average_pnl']:.4f}")
                            st.metric("Текущая позиция", performance['current_position'])
                        
                    else:
                        st.warning("⚠️ Во время тестирования не было выполнено сделок")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при тестировании: {e}")
    else:
        st.warning("⚠️ Сначала обучите агента для этой монеты")

# Показ результатов
if st.session_state.get('show_results', False):
    st.markdown("---")
    st.header("📊 Результаты всех агентов")
    
    if st.session_state.agents:
        # Создаем сводную таблицу
        results_data = []
        for coin, agent in st.session_state.agents.items():
            performance = agent.get_performance_summary()
            if 'total_trades' in performance:
                results_data.append({
                    'Монета': coin,
                    'Всего сделок': performance['total_trades'],
                    'Успешных сделок': performance['successful_trades'],
                    'Процент успеха': f"{performance['success_rate']:.1%}",
                    'Общий P&L': f"{performance['total_pnl']:.4f}",
                    'Средний P&L': f"{performance['average_pnl']:.4f}",
                    'Текущая позиция': performance['current_position']
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # График сравнения
            fig = px.bar(
                results_df, 
                x='Монета', 
                y='Общий P&L',
                title="Сравнение P&L по монетам",
                color='Процент успеха'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
    else:
        st.info("Нет обученных агентов")
    
    st.session_state.show_results = False

# Нижняя панель с информацией
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Что делает система")
    st.markdown("""
    - **Адаптивное обучение** на ошибках
    - **Ансамбль моделей** LSTM
    - **Технические индикаторы** (RSI, MACD, BB)
    - **Управление рисками** (стоп-лосс, тейк-профит)
    """)

with col2:
    st.subheader("🔧 Как использовать")
    st.markdown("""
    1. Выберите монету в боковой панели
    2. Настройте параметры модели
    3. Нажмите 'Обучить агента'
    4. Запустите Sandbox для тестирования
    """)

with col3:
    st.subheader("⚠️ Важно")
    st.markdown("""
    - Это **образовательный проект**
    - Не используйте для реальной торговли
    - Всегда тестируйте на исторических данных
    - Мониторьте производительность
    """)

# Футер
st.markdown("---")
st.markdown("🚀 **Trading Agent Training System** | Создано для обучения и экспериментов")
st.markdown("📧 Для вопросов и предложений обращайтесь к разработчику")

# Автоматическое обновление каждые 30 секунд в Sandbox режиме
if st.session_state.sandbox_mode:
    time.sleep(30)
    st.rerun()
