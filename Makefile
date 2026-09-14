# The only cross-language entry point.

API_DIR := services/api
WEB_DIR := apps/web

.DEFAULT_GOAL := help
.PHONY: help up down logs api web install test test-all test-ci lint fmt migrate revision \
	eval eval-ingest eval-vector record-fixtures

help: ## List targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start the whole stack and wait for it to be healthy
	docker compose up -d --build --wait
	@echo
	@echo "  dashboard  http://localhost:$${WEB_PORT:-3000}"
	@echo "  api        http://localhost:$${API_PORT:-8000}/health/deps"

down: ## Stop everything and remove volumes
	docker compose down -v

logs: ## Tail logs from every service
	docker compose logs -f

install: ## Install both toolchains
	cd $(API_DIR) && uv sync
	cd $(WEB_DIR) && pnpm install

api: ## Run FastAPI on the host with reload (needs `make up` for Postgres/Redis)
	cd $(API_DIR) && uv run uvicorn prism.main:app --reload --port $${API_PORT:-8000}

web: ## Run the Next.js dev server
	cd $(WEB_DIR) && pnpm dev

test: ## Unit tests — no network, no database, no paid call
	cd $(API_DIR) && uv run pytest -m "not integration"

test-all: ## Unit + integration tests (needs `make up` and a local Ollama)
	cd $(API_DIR) && uv run pytest

test-ci: ## Exactly what CI runs — no live model
	cd $(API_DIR) && uv run pytest -m "not ollama"

record-fixtures: ## Re-record paid-provider cassettes (local only, needs real keys)
	cd $(API_DIR) && uv run python ../../scripts/record_fixtures.py $(names)

eval-ingest: ## Ingest the eval corpus into its collection (needs `make up` + Ollama)
	cd $(API_DIR) && uv run python -m prism.eval ingest

eval: ## Golden set vs hybrid retrieval — recall@k (needs `make eval-ingest`)
	cd $(API_DIR) && uv run python -m prism.eval recall --json eval/runs/latest.json

eval-vector: ## Same golden set against the naive baseline retriever
	cd $(API_DIR) && uv run python -m prism.eval recall --retriever vector --json eval/runs/vector.json

lint: ## ruff + mypy + tsc
	cd $(API_DIR) && uv run ruff check . && uv run ruff format --check . && uv run mypy
	cd $(WEB_DIR) && pnpm typecheck

fmt: ## Apply formatting fixes
	cd $(API_DIR) && uv run ruff check --fix . && uv run ruff format .

migrate: ## Apply migrations to the running database
	cd $(API_DIR) && uv run alembic upgrade head

revision: ## Create a migration: make revision m="add cache entries"
	cd $(API_DIR) && uv run alembic revision -m "$(m)"
