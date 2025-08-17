# Makefile для системы обучения торговых агентов

.PHONY: help install test run demo clean docs

# Переменные
PYTHON = python3
PIP = pip3
PYTEST = pytest

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Показать справку по командам
	@echo "$(GREEN)🚀 Система обучения торговых агентов для криптовалюты$(NC)"
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "$(GREEN)📦 Устанавливаем зависимости...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Зависимости установлены$(NC)"

install-dev: ## Установить зависимости для разработки
	@echo "$(GREEN)🔧 Устанавливаем зависимости для разработки...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Зависимости для разработки установлены$(NC)"

test: ## Запустить тесты
	@echo "$(GREEN)🧪 Запускаем тесты...$(NC)"
	$(PYTEST) tests/ -v --cov=crypto_trading_agent --cov=multi_agent_ensemble
	@echo "$(GREEN)✅ Тесты завершены$(NC)"

run: ## Запустить основную систему
	@echo "$(GREEN)🚀 Запускаем систему обучения торговых агентов...$(NC)"
	$(PYTHON) run_trading_system.py

demo: ## Запустить демонстрационный режим
	@echo "$(GREEN)🎯 Запускаем демонстрационный режим...$(NC)"
	$(PYTHON) example_usage.py

notebook: ## Запустить Jupyter notebook
	@echo "$(GREEN)📓 Запускаем Jupyter notebook...$(NC)"
	jupyter notebook train_agents_new.ipynb

gui: ## Запустить GUI (Streamlit)
	@echo "$(GREEN)🖥️  Запускаем GUI (Streamlit)...$(NC)"
	PYTHONPATH=$(shell pwd) poetry run streamlit run src/gui/app.py

ensemble: ## Запустить ансамбль агентов
	@echo "$(GREEN)🔄 Запускаем ансамбль агентов...$(NC)"
	$(PYTHON) multi_agent_ensemble.py

single-agent: ## Обучить одного агента
	@echo "$(GREEN)🤖 Обучаем одного агента...$(NC)"
	$(PYTHON) - <<-'PY'
	from crypto_trading_agent import train_crypto_trading_agent, test_agent_trading
	agent = train_crypto_trading_agent('BTC')
	if agent:
	    test_results = test_agent_trading(agent, test_period_days=3)
	    print('✅ Агент обучен и протестирован')
	else:
	    print('❌ Ошибка обучения агента')
	PY

clean: ## Очистить временные файлы
	@echo "$(YELLOW)🧹 Очищаем временные файлы...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "*.json" -name "*state*" -delete
	find . -type f -name "*.pth" -delete
	@echo "$(GREEN)✅ Очистка завершена$(NC)"

clean-models: ## Очистить обученные модели
	@echo "$(YELLOW)🗑️  Очищаем обученные модели...$(NC)"
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	find . -type f -name "*.ckpt" -delete
	@echo "$(GREEN)✅ Модели очищены$(NC)"

docs: ## Создать документацию
	@echo "$(GREEN)📚 Создаем документацию...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✅ Документация создана$(NC)"

lint: ## Проверить код линтером
	@echo "$(GREEN)🔍 Проверяем код линтером...$(NC)"
	flake8 crypto_trading_agent.py multi_agent_ensemble.py example_usage.py
	black --check crypto_trading_agent.py multi_agent_ensemble.py example_usage.py
	@echo "$(GREEN)✅ Проверка кода завершена$(NC)"

format: ## Форматировать код
	@echo "$(GREEN)✨ Форматируем код...$(NC)"
	black crypto_trading_agent.py multi_agent_ensemble.py example_usage.py
	@echo "$(GREEN)✅ Код отформатирован$(NC)"

check-deps: ## Проверить зависимости
	@echo "$(GREEN)🔍 Проверяем зависимости...$(NC)"
	$(PIP) check
	@echo "$(GREEN)✅ Зависимости в порядке$(NC)"

update-deps: ## Обновить зависимости
	@echo "$(YELLOW)🔄 Обновляем зависимости...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✅ Зависимости обновлены$(NC)"

setup: install ## Полная настройка системы
	@echo "$(GREEN)🎯 Настройка системы завершена!$(NC)"
	@echo "$(YELLOW)Теперь вы можете:$(NC)"
	@echo "  - Запустить систему: make run"
	@echo "  - Запустить демо: make demo"
	@echo "  - Обучить одного агента: make single-agent"
	@echo "  - Запустить notebook: make notebook"

# Команды для разработки
dev-setup: install-dev lint test ## Настройка для разработки
	@echo "$(GREEN)🔧 Настройка для разработки завершена!$(NC)"

# Команды для продакшена
prod-setup: install check-deps ## Настройка для продакшена
	@echo "$(GREEN)🚀 Настройка для продакшена завершена!$(NC)"

# Информационные команды
info: ## Показать информацию о системе
	@echo "$(GREEN)📋 Информация о системе:$(NC)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Не установлен')"
	@echo "  CUDA доступна: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Неизвестно')"
	@echo "  Доступные GPU: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"

status: ## Показать статус системы
	@echo "$(GREEN)📊 Статус системы:$(NC)"
	@echo "  Файлы агентов: $(shell ls -1 *.py 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  Notebooks: $(shell ls -1 *.ipynb 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  Логи: $(shell ls -1 *.log 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  Состояния: $(shell ls -1 *state*.json 2>/dev/null | wc -l | tr -d ' ')"

# Команды для мониторинга
logs: ## Показать логи
	@echo "$(GREEN)📝 Последние логи:$(NC)"
	@tail -n 20 trading_system.log 2>/dev/null || echo "Логи не найдены"

monitor: ## Мониторинг в реальном времени
	@echo "$(GREEN)👀 Запускаем мониторинг...$(NC)"
	@tail -f trading_system.log 2>/dev/null || echo "Логи не найдены"

# Команды для резервного копирования
backup: ## Создать резервную копию
	@echo "$(GREEN)💾 Создаем резервную копию...$(NC)"
	@tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz *.py *.ipynb *.md *.txt 2>/dev/null || echo "Ошибка создания резервной копии"
	@echo "$(GREEN)✅ Резервная копия создана$(NC)"

restore: ## Восстановить из резервной копии
	@echo "$(YELLOW)🔄 Восстанавливаем из резервной копии...$(NC)"
	@ls -1 backup_*.tar.gz 2>/dev/null | head -1 | xargs -I {} tar -xzf {} || echo "Резервная копия не найдена"
	@echo "$(GREEN)✅ Восстановление завершено$(NC)"

# Справка по командам
commands: ## Показать все доступные команды
	@echo "$(GREEN)📚 Все доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*$$/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Команда по умолчанию
.DEFAULT_GOAL := help


