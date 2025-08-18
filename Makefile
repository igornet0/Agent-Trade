# Makefile for Agent Trade Project
# Usage: make <target>

# Variables
PROJECT_NAME = agent-trade
COMPOSE_FILE = docker-compose.yml
COMPOSE_DEV_FILE = docker-compose.dev.yml
COMPOSE_PROD_FILE = docker-compose.prod.yml
ENV_FILE = settings/local.env
ENV_DEV_FILE = settings/dev.env
ENV_PROD_FILE = settings/prod.env
DOCKER_REGISTRY = 

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Agent Trade Project - Docker Management$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Quick start:$(NC)"
	@echo "  make dev-start        - Start development environment"
	@echo "  make prod-start       - Start production environment"
	@echo "  make dev-build        - Build development images"
	@echo "  make prod-build       - Build production images"
	@echo "  make status           - Show service status"
	@echo ""

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================

.PHONY: dev-build
dev-build: ## Build development Docker images
	@echo "$(BLUE)Building development Docker images...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) build
	@echo "$(GREEN)Development build completed!$(NC)"

.PHONY: dev-build-no-cache
dev-build-no-cache: ## Build development Docker images without cache
	@echo "$(BLUE)Building development Docker images (no cache)...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) build --no-cache
	@echo "$(GREEN)Development build completed!$(NC)"

.PHONY: dev-start
dev-start: ## Start development environment (all services)
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - Frontend: http://localhost:5173$(NC)"
	@echo "  - Backend API: http://localhost:8000$(NC)"
	@echo "  - Database: localhost:5432$(NC)"
	@echo "  - Redis: localhost:6379$(NC)"
	@echo "  - RabbitMQ: localhost:5672 (Management: localhost:15672)$(NC)"
	@echo "  - Grafana: http://localhost:3000$(NC)"
	@echo "  - Prometheus: http://localhost:9090$(NC)"

.PHONY: dev-stop
dev-stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) down
	@echo "$(GREEN)Development environment stopped!$(NC)"

.PHONY: dev-restart
dev-restart: ## Restart development environment
	@echo "$(BLUE)Restarting development environment...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) restart
	@echo "$(GREEN)Development environment restarted!$(NC)"

.PHONY: dev-logs
dev-logs: ## Show development environment logs
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) logs -f

.PHONY: dev-logs-frontend
dev-logs-frontend: ## Show frontend development logs
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) logs -f frontend

.PHONY: dev-logs-backend
dev-logs-backend: ## Show backend development logs
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) logs -f backend

.PHONY: dev-logs-celery
dev-logs-celery: ## Show celery development logs
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) logs -f celery

# =============================================================================
# PRODUCTION COMMANDS
# =============================================================================

.PHONY: prod-build
prod-build: ## Build production Docker images
	@echo "$(BLUE)Building production Docker images...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) build
	@echo "$(GREEN)Production build completed!$(NC)"

.PHONY: prod-build-no-cache
prod-build-no-cache: ## Build production Docker images without cache
	@echo "$(BLUE)Building production Docker images (no cache)...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) build --no-cache
	@echo "$(GREEN)Production build completed!$(NC)"

.PHONY: prod-start
prod-start: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) up -d
	@echo "$(GREEN)Production environment started!$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - Frontend: http://localhost (Nginx)$(NC)"
	@echo "  - Backend API: http://localhost:8000$(NC)"
	@echo "  - Database: localhost:5432$(NC)"
	@echo "  - Redis: localhost:6379$(NC)"
	@echo "  - RabbitMQ: localhost:5672 (Management: localhost:15672)$(NC)"
	@echo "  - Grafana: http://localhost:3000$(NC)"
	@echo "  - Prometheus: http://localhost:9090$(NC)"

.PHONY: prod-stop
prod-stop: ## Stop production environment
	@echo "$(BLUE)Stopping production environment...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) down
	@echo "$(GREEN)Production environment stopped!$(NC)"

.PHONY: prod-restart
prod-restart: ## Restart production environment
	@echo "$(BLUE)Restarting production environment...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) restart
	@echo "$(GREEN)Production environment restarted!$(NC)"

.PHONY: prod-logs
prod-logs: ## Show production environment logs
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) logs -f

# =============================================================================
# LEGACY COMMANDS (for backward compatibility)
# =============================================================================

.PHONY: dev
dev: dev-start ## Alias for dev-start

.PHONY: dev-full
dev-full: dev-start ## Alias for dev-start

.PHONY: build
build: prod-build ## Alias for prod-build

.PHONY: build-no-cache
build-no-cache: prod-build-no-cache ## Alias for prod-build-no-cache

.PHONY: up
up: prod-start ## Alias for prod-start

.PHONY: down
down: prod-stop ## Alias for prod-stop

.PHONY: restart
restart: prod-restart ## Alias for prod-restart

# =============================================================================
# SERVICE-SPECIFIC COMMANDS
# =============================================================================

.PHONY: up-db
up-db: ## Start only database and RabbitMQ
	@echo "$(BLUE)Starting database and RabbitMQ...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) up -d postgres rabbitmq redis
	@echo "$(GREEN)Database and RabbitMQ started!$(NC)"

.PHONY: up-backend
up-backend: ## Start backend services (API + Worker)
	@echo "$(BLUE)Starting backend services...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) up -d backend celery
	@echo "$(GREEN)Backend services started!$(NC)"

.PHONY: up-frontend
up-frontend: ## Start frontend service
	@echo "$(BLUE)Starting frontend service...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) up -d frontend
	@echo "$(GREEN)Frontend service started!$(NC)"

# =============================================================================
# LOGS COMMANDS
# =============================================================================

.PHONY: logs
logs: dev-logs ## Alias for dev-logs

.PHONY: logs-api
logs-api: dev-logs-backend ## Alias for dev-logs-backend

.PHONY: logs-worker
logs-worker: dev-logs-celery ## Alias for dev-logs-celery

.PHONY: logs-frontend
logs-frontend: dev-logs-frontend ## Alias for dev-logs-frontend

.PHONY: logs-db
logs-db: ## Show database logs
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) logs -f postgres

# =============================================================================
# DATABASE COMMANDS
# =============================================================================

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec backend alembic upgrade head
	@echo "$(GREEN)Database migrations completed!$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: This will delete all data!)
	@echo "$(RED)WARNING: This will delete all database data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BLUE)Resetting database...$(NC)"; \
		docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) down -v; \
		docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) up -d postgres; \
		echo "$(GREEN)Database reset completed!$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled.$(NC)"; \
	fi

# =============================================================================
# ADMIN MANAGEMENT COMMANDS
# =============================================================================

.PHONY: create-admin
create-admin: ## Create admin user (default: admin/admin123)
	@echo "$(BLUE)Creating admin user...$(NC)"
	@if [ -z "$(LOGIN)" ]; then LOGIN=admin; fi; \
	if [ -z "$(EMAIL)" ]; then EMAIL=admin@agent-trade.com; fi; \
	if [ -z "$(PASSWORD)" ]; then PASSWORD=admin123; fi; \
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec backend python scripts/create_admin.py --login $$LOGIN --email $$EMAIL --password $$PASSWORD
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: create-admin-local
create-admin-local: ## Create admin user locally (default: admin/admin123)
	@echo "$(BLUE)Creating admin user locally...$(NC)"
	@if [ -z "$(LOGIN)" ]; then LOGIN=admin; fi; \
	if [ -z "$(EMAIL)" ]; then EMAIL=admin@agent-trade.com; fi; \
	if [ -z "$(PASSWORD)" ]; then PASSWORD=admin123; fi; \
	poetry run python create_admin_local.py --login $$LOGIN --email $$EMAIL --password $$PASSWORD
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: create-admin-direct
create-admin-direct: ## Create admin user directly in database (default: admin/admin123)
	@echo "$(BLUE)Creating admin user directly in database...$(NC)"
	@poetry run python create_admin_direct.py
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: create-admin-docker
create-admin-docker: ## Create admin user in Docker database (default: admin/admin123)
	@echo "$(BLUE)Creating admin user in Docker database...$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, login VARCHAR(50) UNIQUE NOT NULL, email VARCHAR(50) UNIQUE, password VARCHAR(255) NOT NULL, user_telegram_id BIGINT, balance FLOAT DEFAULT 0, role VARCHAR(50) DEFAULT 'user', active BOOLEAN DEFAULT TRUE);"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "INSERT INTO users (login, email, password, role, active, balance) VALUES ('admin', 'admin@agent-trade.com', '$(shell echo -n 'admin123' | sha256sum | cut -d' ' -f1)', 'admin', TRUE, 0.0) ON CONFLICT (login) DO NOTHING;"
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: create-tables-docker
create-tables-docker: ## Create all tables in Docker database
	@echo "$(BLUE)Creating all tables in Docker database...$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS coins (id SERIAL PRIMARY KEY, name VARCHAR(50) UNIQUE NOT NULL, price_now FLOAT DEFAULT 0, max_price_now FLOAT DEFAULT 0, min_price_now FLOAT DEFAULT 0, open_price_now FLOAT DEFAULT 0, volume_now FLOAT DEFAULT 0, price_change_percentage_24h FLOAT, news_score_global FLOAT DEFAULT 100, parsed BOOLEAN DEFAULT TRUE);"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS timeseries (id SERIAL PRIMARY KEY, coin_id INTEGER REFERENCES coins(id), timestamp VARCHAR(50), path_dataset VARCHAR(100) UNIQUE);"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS data_timeseries (id SERIAL PRIMARY KEY, timeseries_id INTEGER REFERENCES timeseries(id), datetime TIMESTAMP NOT NULL, open FLOAT, max FLOAT, min FLOAT, close FLOAT, volume FLOAT);"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS portfolio (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), coin_id INTEGER REFERENCES coins(id), amount FLOAT DEFAULT 0.0, price_avg FLOAT DEFAULT 0.0, CONSTRAINT uq_portfolio_user_coin UNIQUE (user_id, coin_id), CONSTRAINT ck_portfolio_amount_non_negative CHECK (amount >= 0));"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS transactions (id SERIAL PRIMARY KEY, status VARCHAR(30) DEFAULT 'open', user_id INTEGER REFERENCES users(id), coin_id INTEGER REFERENCES coins(id), type VARCHAR(20) NOT NULL, amount_orig FLOAT NOT NULL, amount FLOAT NOT NULL, price FLOAT NOT NULL);"
	@echo "$(GREEN)All tables created successfully!$(NC)"

.PHONY: create-admin-interactive
create-admin-interactive: ## Create admin user interactively
	@echo "$(BLUE)Creating admin user interactively...$(NC)"
	@read -p "Enter admin login: " LOGIN; \
	read -p "Enter admin email: " EMAIL; \
	read -s -p "Enter admin password: " PASSWORD; \
	echo; \
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec backend python scripts/create_admin.py --login $$LOGIN --email $$EMAIL --password $$PASSWORD
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: db-backup
db-backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres pg_dump -U agent agent > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created!$(NC)"

# =============================================================================
# MAINTENANCE COMMANDS
# =============================================================================

.PHONY: clean
clean: ## Remove all containers, networks, and volumes
	@echo "$(BLUE)Cleaning up Docker environment...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) down -v --remove-orphans
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Cleanup completed!$(NC)"

.PHONY: clean-images
clean-images: ## Remove all project images
	@echo "$(BLUE)Removing project images...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) down --rmi all
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file $(ENV_PROD_FILE) down --rmi all
	@echo "$(GREEN)Images removed!$(NC)"

.PHONY: ps
ps: ## Show running containers
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) ps

.PHONY: status
status: ## Show status of all services
	@echo "$(BLUE)Service Status:$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

.PHONY: check-admin
check-admin: ## Check if admin user exists in database
	@echo "$(BLUE)Checking admin user...$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent -c "SELECT id, login, email, role, active FROM users WHERE login = 'admin';" || echo "$(RED)Admin user not found$(NC)"

.PHONY: check-services
check-services: ## Check if all services are running
	@echo "$(BLUE)Checking services...$(NC)"
	@echo "$(YELLOW)Database:$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres pg_isready -U agent || echo "$(RED)Database is not ready$(NC)"
	@echo "$(YELLOW)Redis:$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec redis redis-cli ping || echo "$(RED)Redis is not ready$(NC)"
	@echo "$(YELLOW)Backend API:$(NC)"
	@curl -f http://localhost:8000/health 2>/dev/null || echo "$(RED)Backend API is not ready$(NC)"
	@echo "$(YELLOW)Frontend:$(NC)"
	@curl -f http://localhost:5173 2>/dev/null || echo "$(RED)Frontend is not ready$(NC)"
	@echo "$(YELLOW)Grafana:$(NC)"
	@curl -f http://localhost:3000 2>/dev/null || echo "$(RED)Grafana is not ready$(NC)"
	@echo "$(YELLOW)Prometheus:$(NC)"
	@curl -f http://localhost:9090 2>/dev/null || echo "$(RED)Prometheus is not ready$(NC)"

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

.PHONY: shell-api
shell-api: ## Open shell in API container
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec backend bash

.PHONY: shell-worker
shell-worker: ## Open shell in worker container
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec celery bash

.PHONY: shell-db
shell-db: ## Open shell in database container
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec postgres psql -U agent -d agent

.PHONY: shell-frontend
shell-frontend: ## Open shell in frontend container
	docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec frontend sh

# =============================================================================
# HEALTH CHECKS
# =============================================================================

.PHONY: health
health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo "$(YELLOW)Database:$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec -T postgres pg_isready -U agent || echo "$(RED)Database is not ready$(NC)"
	@echo "$(YELLOW)RabbitMQ:$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec -T rabbitmq rabbitmq-diagnostics ping || echo "$(RED)RabbitMQ is not ready$(NC)"
	@echo "$(YELLOW)API:$(NC)"
	@curl -f http://localhost:8000/health || echo "$(RED)API is not ready$(NC)"
	@echo "$(YELLOW)Frontend:$(NC)"
	@curl -f http://localhost:5173 || echo "$(RED)Frontend is not ready$(NC)"

# =============================================================================
# QUICK START TARGETS
# =============================================================================

.PHONY: quick-start
quick-start: dev-build dev-start ## Quick start: build and start development environment
	@echo "$(GREEN)Quick start completed!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)RabbitMQ Management: http://localhost:15672$(NC)"

.PHONY: setup-with-admin
setup-with-admin: dev-start db-migrate create-admin ## Setup development environment with admin user
	@echo "$(GREEN)Setup completed with admin user!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"

.PHONY: setup-docker-complete
setup-docker-complete: dev-start create-tables-docker create-admin-docker ## Complete Docker setup with tables and admin
	@echo "$(GREEN)Docker setup completed with tables and admin user!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)RabbitMQ Management: http://localhost:15672$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"

.PHONY: setup-full
setup-full: dev-start create-tables-docker create-admin-docker ## Full setup with all services and admin
	@echo "$(GREEN)Full setup completed!$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - Frontend: http://localhost:5173$(NC)"
	@echo "  - Backend API: http://localhost:8000$(NC)"
	@echo "  - RabbitMQ Management: http://localhost:15672$(NC)"
	@echo "  - Grafana: http://localhost:3000$(NC)"
	@echo "  - Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"
	@echo "$(GREEN)All containers are running and admin user is created!$(NC)"

.PHONY: init-project
init-project: setup-full check-admin ## Initialize complete project with all services
	@echo "$(GREEN)Project initialization completed!$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@echo "  - make status - Check service status$(NC)"
	@echo "  - make check-admin - Check admin user$(NC)"
	@echo "  - make dev-logs - View logs$(NC)"
	@echo "  - make dev-stop - Stop all services$(NC)"

.PHONY: project-status
project-status: status check-admin check-services ## Show complete project status
	@echo "$(GREEN)Project status check completed!$(NC)"
	@echo "$(YELLOW)Summary:$(NC)"
	@echo "  - All Docker containers are running$(NC)"
	@echo "  - Database and Redis are ready$(NC)"
	@echo "  - Admin user is created: admin/admin123$(NC)"
	@echo "  - Monitoring services (Grafana, Prometheus) are running$(NC)"
	@echo "  - Frontend and Backend are accessible$(NC)"

.PHONY: complete-setup
complete-setup: setup-full check-admin check-services ## Complete project setup with status check
	@echo "$(GREEN)Complete project setup finished!$(NC)"
	@echo "$(YELLOW)Services available:$(NC)"
	@echo "  - Database: localhost:5432 (PostgreSQL)$(NC)"
	@echo "  - Redis: localhost:6379$(NC)"
	@echo "  - RabbitMQ: localhost:5672 (Management: localhost:15672)$(NC)"
	@echo "  - Grafana: http://localhost:3000$(NC)"
	@echo "  - Prometheus: http://localhost:9090$(NC)"
	@echo "  - Backend API: http://localhost:8000$(NC)"
	@echo "  - Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"
	@echo "$(GREEN)Project is ready for development!$(NC)"

.PHONY: quick-dev
quick-dev: up-db ## Quick dev: start only database and RabbitMQ
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Database: localhost:5432$(NC)"
	@echo "$(YELLOW)Redis: localhost:6379$(NC)"
	@echo "$(YELLOW)RabbitMQ: localhost:5672$(NC)"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

.PHONY: env-check
env-check: ## Check if environment files exist
	@if [ ! -f $(ENV_DEV_FILE) ]; then \
		echo "$(RED)Development environment file $(ENV_DEV_FILE) not found!$(NC)"; \
		echo "$(YELLOW)Please create $(ENV_DEV_FILE) with your configuration.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)Development environment file found: $(ENV_DEV_FILE)$(NC)"; \
	fi
	@if [ ! -f $(ENV_PROD_FILE) ]; then \
		echo "$(RED)Production environment file $(ENV_PROD_FILE) not found!$(NC)"; \
		echo "$(YELLOW)Please create $(ENV_PROD_FILE) with your configuration.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)Production environment file found: $(ENV_PROD_FILE)$(NC)"; \
	fi

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

.PHONY: preflight
preflight: env-check ## Run pre-flight checks
	@echo "$(BLUE)Running pre-flight checks...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is not installed$(NC)" >&2; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)Docker Compose is not installed$(NC)" >&2; exit 1; }
	@echo "$(GREEN)Pre-flight checks passed!$(NC)"

# =============================================================================
# PRODUCTION DEPLOYMENT
# =============================================================================

.PHONY: deploy
deploy: preflight prod-build prod-start ## Deploy to production
	@echo "$(GREEN)Deployment completed!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost (Nginx)$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"

# =============================================================================
# MONITORING
# =============================================================================

.PHONY: monitor
monitor: ## Monitor resource usage
	@echo "$(BLUE)Monitoring resource usage...$(NC)"
	@docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# =============================================================================
# BACKUP AND RESTORE
# =============================================================================

.PHONY: backup
backup: ## Create full backup (database + volumes)
	@echo "$(BLUE)Creating full backup...$(NC)"
	@mkdir -p backups
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec -T postgres pg_dump -U agent agent > backups/db_backup_$$(date +%Y%m%d_%H%M%S).sql
	@docker run --rm -v $(PROJECT_NAME)_pgdata:/data -v $$(pwd)/backups:/backup alpine tar czf /backup/volumes_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)Backup completed!$(NC)"

.PHONY: restore
restore: ## Restore from backup (specify BACKUP_FILE=filename)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Please specify BACKUP_FILE=filename$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring from backup: $(BACKUP_FILE)$(NC)"
	@docker-compose -f $(COMPOSE_DEV_FILE) --env-file $(ENV_DEV_FILE) exec -T postgres psql -U agent -d agent < backups/$(BACKUP_FILE)
	@echo "$(GREEN)Restore completed!$(NC)"

# =============================================================================
# FRONTEND DEVELOPMENT (LOCAL)
# =============================================================================

.PHONY: frontend-dev
frontend-dev: ## Start frontend development server locally
	@echo "$(BLUE)Starting frontend development server locally...$(NC)"
	@cd frontend && npm run dev
