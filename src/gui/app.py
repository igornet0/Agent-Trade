import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import asyncio
import torch

# sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# core and models
from core import data_helper
from Dataset import DatasetTimeseries, LoaderTimeLine
from MMM import AgentManager
from train_models.loader import Loader as TrainLoader

from core.database.engine import db_helper, set_db_helper, initialize_db_helper
from core.database.orm.agents import orm_get_agents, orm_set_active_version
from core.database.orm.strategies import orm_list_strategies_for_user
from core.database.orm.ml_models import (
    orm_get_ml_models,
    orm_add_ml_model,
    orm_add_model_stat,
)


st.set_page_config(page_title="Agent Trade – Manager", page_icon="🧠", layout="wide")
st.title("🧠 Agent Manager – Обучение, Оценка, База моделей и Стратегии")
st.markdown("---")


# -------- Helpers --------
@st.cache_data(show_spinner=False)
def list_coins() -> list[str]:
    try:
        return list(data_helper.coin_list)
    except Exception:
        coins = set()
        for p in data_helper.get_path("processed", timetravel="5m"):
            coins.add(p.parent.name)
        return sorted(coins)


@st.cache_data(show_spinner=True)
def load_coin_dataframe(coin: str, timeframe: str = "5m") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in data_helper.get_path("processed", coin=coin, timetravel=timeframe):
        try:
            dt = DatasetTimeseries(str(p), timetravel=timeframe)
            df = dt.get_dataset().copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            for c in ("open", "close", "max", "min", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["datetime", "open", "close", "max", "min", "volume"])  # пустой
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["datetime", "open", "close", "max", "min", "volume"]).sort_values("datetime").drop_duplicates("datetime")
    return data.reset_index(drop=True)


def render_price_chart(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Нет данных для отображения")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["max"], low=df["min"], close=df["close"], name="OHLC"
    ))
    fig.update_layout(title=title, height=420, xaxis_title="Время", yaxis_title="Цена")
    st.plotly_chart(fig, use_container_width=True)


def build_timeline_loaders(coin_list: list[str], timeframe: str, window: int) -> list[LoaderTimeLine]:
    loaders: list[LoaderTimeLine] = []
    for coin in coin_list:
        df = load_coin_dataframe(coin, timeframe)
        if df.empty:
            continue
        dt = DatasetTimeseries(df, timetravel=timeframe)
        loaders.append(dt.get_time_line_loader(time_line_size=window))
    return loaders


async def ensure_db():
    if not db_helper:
        await set_db_helper()


async def fetch_agents_from_db(agent_type: str | None = None, timeframe: str | None = None):
    # создаём локальный AsyncEngine в рамках текущего event loop
    local_db = await initialize_db_helper()
    try:
        async with local_db.get_session() as session:
            agents = await orm_get_agents(session, type_agent=agent_type)
            if agents and timeframe:
                agents = [a for a in agents if a.get("timeframe") == timeframe]
            return agents or []
    finally:
        await local_db.dispose()


async def fetch_strategies(user_id: int = 1):
    local_db = await initialize_db_helper()
    try:
        async with local_db.get_session() as session:
            return await orm_list_strategies_for_user(session, user_id)
    finally:
        await local_db.dispose()


async def fetch_ml_models(type_filter: str | None = None):
    local_db = await initialize_db_helper()
    try:
        async with local_db.get_session() as session:
            return await orm_get_ml_models(session, type=type_filter)
    finally:
        await local_db.dispose()


# -------- Sidebar --------
coins = list_coins()
all_agent_types = list(AgentManager.type_agents.keys())

with st.sidebar:
    st.header("⚙️ Параметры")
    agent_type = st.selectbox("Тип агента", options=all_agent_types, index=0)
    timeframe = st.selectbox("Таймфрейм", options=["5m", "15m", "30m", "1H", "4H", "1D"], index=0)
    selected_coins = st.multiselect("Монеты", options=coins, default=coins[:3])
    window = st.slider("Окно последовательности", 20, 200, 50, 5)

    st.subheader("Обучение")
    epochs = st.slider("Эпохи", 1, 100, 10)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    lr = st.selectbox("Learning rate", [1e-3, 5e-4, 1e-4], index=0)
    weight_decay = st.selectbox("Weight decay", [1e-2, 1e-3, 1e-4], index=1)
    patience = st.slider("Patience", 3, 20, 7)

    st.markdown("---")
    train_btn = st.button("🚀 Обучить выбранного агента", type="primary")


tab_data, tab_db, tab_train, tab_eval, tab_models, tab_strategy, tab_sandbox = st.tabs(["Данные", "Агенты (БД)", "Обучение", "Оценка", "Модели (БД)", "Стратегии", "Песочница"])


with tab_data:
    coin_preview = st.selectbox("Монета для предпросмотра", options=coins, index=0, key="preview_coin")
    df = load_coin_dataframe(coin_preview, timeframe)
    st.caption(f"Записей: {len(df)}")
    render_price_chart(df.tail(3000), f"{coin_preview} – {timeframe}")
    st.dataframe(df.tail(10), use_container_width=True)


with tab_db:
    st.subheader("Сохранённые агенты")
    try:
        agents_list = asyncio.run(fetch_agents_from_db(agent_type=agent_type, timeframe=timeframe))
        if agents_list:
            # Преобразуем сложные поля к строкам, чтобы избежать ошибок Arrow
            df_agents = pd.DataFrame(agents_list)
            if "features" in df_agents.columns:
                df_agents["features"] = df_agents["features"].apply(lambda x: str(x))
            st.dataframe(df_agents, use_container_width=True)
            # Управление активной версией
            col_a, col_b = st.columns([2, 1])
            with col_a:
                sel = st.selectbox("Выбрать агента для активации", options=[f"{a['id']} | {a['name']} v{a['version']}" for a in agents_list])
            with col_b:
                if st.button("Сделать активным"):
                    agent_id = int(sel.split(" |")[0])
                    try:
                        asyncio.run(_ := (lambda aid=agent_id: (asyncio.run(ensure_db()), None))())
                    except RuntimeError:
                        pass
                    async def _activate(aid: int):
                        await ensure_db()
                        async with db_helper.get_session() as session:
                            await orm_set_active_version(session, aid)
                    try:
                        asyncio.run(_activate(agent_id))
                        st.success("Версия активирована")
                    except Exception as e:
                        st.error(f"Ошибка активации: {e}")
        else:
            st.info("В БД нет сохранённых агентов для выбранных фильтров")
    except Exception as e:
        st.warning(f"БД недоступна: {e}")


with tab_train:
    st.subheader("Обучение через AgentManager")
    if not selected_coins:
        st.info("Выберите хотя бы одну монету слева")
    else:
        if train_btn:
            with st.spinner("Готовим данные и агент..."):
                try:
                    # Конфиг формируется из моделей в data_helper при необходимости, здесь берём дефолт
                    config = {
                        "agents": [
                            {
                                "type": agent_type,
                                "name": f"{agent_type}_{timeframe}",
                                "indecaters": {},
                                "timetravel": timeframe,
                                "model_parameters": {"seq_len": window, "pred_len": 5},
                                "data_normalize": True,
                                "mod": "train",
                            }
                        ]
                    }

                    am = AgentManager(config=config, count_agents=1)
                    agent = am.get_agents()

                    # Предполагаем путь, по которому будет сохранена модель
                    try:
                        pth_path = str(agent.get_filename_pth())
                    except Exception:
                        pth_path = None

                    loaders = build_timeline_loaders(selected_coins, timeframe, window)
                    if not loaders:
                        st.error("Не удалось подготовить данные. Проверьте data/processed")
                    else:
                        trainer = TrainLoader(agent_type=agent_type, model_type="MMM")
                        # Переопределяем гиперпараметры через временный конфиг
                        # (Loader читает из data_helper конфиг по имени, поэтому обучим напрямую _train_single_agent)
                        history = trainer._train_single_agent(
                            agent=agent,
                            loaders=loaders,
                            epochs=epochs,
                            batch_size=batch_size,
                            base_lr=lr,
                            weight_decay=weight_decay,
                            patience=patience,
                            mixed=True,
                            mixed_precision=False,
                        )
                        st.success("✅ Обучение завершено")
                        st.line_chart(pd.Series(history, name="loss"))

                        # Сохраняем запись о модели в БД
                        if pth_path:
                            try:
                                async def _save_model_record():
                                    local_db = await initialize_db_helper()
                                    try:
                                        async with local_db.get_session() as session:
                                            await orm_add_ml_model(session, type=agent_type, path_model=pth_path, version=str(getattr(agent, 'version', '0.0.1')))
                                    finally:
                                        await local_db.dispose()
                                asyncio.run(_save_model_record())
                                st.toast("Модель зарегистрирована в БД", icon="✅")
                            except Exception as e:
                                st.warning(f"Не удалось записать модель в БД: {e}")
                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
with tab_eval:
    st.subheader("Оценка моделей/агентов")
    col1, col2 = st.columns(2)
    with col1:
        eval_agent_type = st.selectbox("Тип для оценки", options=all_agent_types, index=0, key="eval_agent_type")
        eval_timeframe = st.selectbox("Таймфрейм", options=["5m", "15m", "30m", "1H", "4H", "1D"], index=0, key="eval_timeframe")
        eval_coins = st.multiselect("Монеты", options=coins, default=coins[:2], key="eval_coins")
        eval_window = st.slider("Окно последовательности", 20, 200, 50, 5, key="eval_window")
    with col2:
        st.caption("Быстрая валидация на выбранных монетах")
        run_eval = st.button("▶️ Запустить оценку")

    if run_eval:
        try:
            loaders = build_timeline_loaders(eval_coins, eval_timeframe, eval_window)
            if not loaders:
                st.error("Нет данных для оценки")
            else:
                config = {
                    "agents": [
                        {
                            "type": eval_agent_type,
                            "name": f"{eval_agent_type}_{eval_timeframe}",
                            "indecaters": {},
                            "timetravel": eval_timeframe,
                            "model_parameters": {"seq_len": eval_window, "pred_len": 5},
                            "data_normalize": True,
                            "mod": "test",
                        }
                    ]
                }
                am = AgentManager(config=config, count_agents=1)
                agent = am.get_agents()
                ds = TrainLoader(agent_type=eval_agent_type, model_type="MMM").load_agent_data(loaders, agent, batch_size=32, mixed=False)
                import torch
                from torch import no_grad
                losses = []
                n_batches = 0
                for (x, y, t) in ds:
                    with no_grad():
                        out = agent.trade([x, t] if t is not None else [x])
                        try:
                            loss = agent.loss_function(out, y)
                            losses.append(float(loss.detach().cpu().item()))
                        except Exception:
                            pass
                        n_batches += 1
                        if n_batches >= 20:
                            break
                if losses:
                    st.success(f"Средний loss (первые {len(losses)} батчей): {sum(losses)/len(losses):.4f}")
                    st.line_chart(pd.Series(losses, name="loss_batch"))
                    # Сохранение метрик в БД
                    with st.expander("Сохранить метрики в БД"):
                        try:
                            models_for_type = asyncio.run(fetch_ml_models(type_filter=eval_agent_type))
                        except Exception:
                            models_for_type = []
                        if models_for_type:
                            selection = st.selectbox(
                                "Записать в модель",
                                options=[f"#{m.id} {m.type} | {m.version} | {m.path_model}" for m in models_for_type],
                            )
                            save_metrics = st.button("💾 Сохранить как test")
                            if save_metrics:
                                try:
                                    model_id = int(selection.split()[0].replace('#', ''))
                                    async def _save_stats(mid: int, loss_avg: float):
                                        local_db = await initialize_db_helper()
                                        try:
                                            async with local_db.get_session() as session:
                                                await orm_add_model_stat(
                                                    session,
                                                    model_id=mid,
                                                    type="test",
                                                    loss=loss_avg,
                                                    accuracy=0.0,
                                                    precision=0.0,
                                                    recall=0.0,
                                                    f1=0.0,
                                                )
                                        finally:
                                            await local_db.dispose()
                                    asyncio.run(_save_stats(model_id, sum(losses)/len(losses)))
                                    st.success("Метрики сохранены")
                                except Exception as e:
                                    st.error(f"Не удалось сохранить метрики: {e}")
                        else:
                            st.info("Нет зарегистрированных моделей для данного типа. Сначала обучите и сохраните запись о модели.")
                else:
                    st.info("Не удалось вычислить метрики для данной конфигурации")
        except Exception as e:
            st.error(f"Ошибка оценки: {e}")

with tab_models:
    st.subheader("ML Модели (БД)")
    try:
        models = asyncio.run(fetch_ml_models())
        if models:
            import pandas as _pd
            df_models = _pd.DataFrame([
                {"id": m.id, "type": m.type, "version": m.version, "path": m.path_model} for m in models
            ])
            st.dataframe(df_models, use_container_width=True)
        else:
            st.info("В БД нет зарегистрированных моделей")
    except Exception as e:
        st.warning(f"БД недоступна: {e}")

with tab_sandbox:
    st.subheader("Песочница (бэктест)")
    st.caption("Минимальный запуск: загрузка агентов из БД и данных из файлов/БД")
    sb_coin = st.text_input("Монета (по умолчанию первая из данных)", value="")
    sb_tf = st.text_input("Таймфрейм (по умолчанию первый)", value="")
    sb_steps = st.slider("Шагов", 10, 500, 100, 10)
    run_sb = st.button("🏁 Запустить (демо)")
    if run_sb:
        try:
            from SandboxApp.sandbox import Sandbox
            sb = Sandbox(db_use=True)
            res = sb.start(coin=sb_coin or None, timeframe=sb_tf or None, max_steps=sb_steps)
            st.write(res)
        except Exception as e:
            st.warning(f"Песочница пока не готова: {e}")


with tab_strategy:
    st.subheader("Стратегии (из БД)")
    try:
        strategies = asyncio.run(fetch_strategies(user_id=1))
        if strategies:
            df_str = pd.DataFrame([{"id": s.id, "name": s.name, "type": s.type, "risk": s.risk, "reward": s.reward} for s in strategies])
            st.dataframe(df_str, use_container_width=True)
        else:
            st.info("Стратегии для пользователя #1 не найдены")
    except Exception as e:
        st.warning(f"БД недоступна: {e}")

st.markdown("---")
st.caption("GUI через AgentManager. Для записи в БД используйте соответствующие ORM-функции.")


