.DEFAULT_GOAL := help
.PHONY: help install data external ingest geo backtest forecast pipeline run lint format test check loadtest up-local down-local clean

COMPOSE := docker compose -f docker/docker-compose.yml --env-file .env

help: ## Показать список целей
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Установка ---
install: ## Установить зависимости (uv)
	uv sync

data: ## Распаковать dataset.zip в data/raw
	mkdir -p data/raw && unzip -o -q dataset.zip -d data/raw

external: ## Скачать внешние данные (погода Open-Meteo, маршруты OSM)
	uv run python scripts/fetch_external.py

# --- Пайплайн ---
ingest: ## Сырые валидации → data/processed/route_hour.parquet
	uv run python -m tram_forecast.ingest

geo: ## Справочник + OSM → artifacts/geo.json
	uv run python -m tram_forecast.geo

backtest: ## Rolling-origin бэктест → artifacts/metrics.json
	uv run python -m tram_forecast.model backtest

forecast: ## Обучение и прогноз → artifacts/forecast.parquet, submission.csv
	uv run python -m tram_forecast.model forecast

pipeline: ingest geo backtest forecast ## Полный пайплайн от сырых данных

# --- Запуск ---
run: ## Запустить API и дашборд локально (http://localhost:8000)
	uv run uvicorn tram_forecast.api:app --host 0.0.0.0 --port 8000 --workers 4

# --- Проверка ---
lint: ## Проверить код ruff
	uv run ruff check src scripts tests

format: ## Отформатировать код ruff
	uv run ruff format src scripts tests

test: ## Запустить тесты
	uv run pytest -q

check: lint test ## lint + test

loadtest: ## Нагрузочный тест запущенного сервиса (URL=http://localhost:8000)
	scripts/loadtest.sh $(or $(URL),http://localhost:8000)

# --- Docker ---
up-local: ## Собрать и запустить контейнер
	$(COMPOSE) up -d --build

down-local: ## Остановить контейнер
	$(COMPOSE) down

# --- Обслуживание ---
clean: ## Удалить кэши
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
