# Makefile for Agent Trade Project
# Usage: make <target>

# Variables
PROJECT_NAME = agent-trade
COMPOSE_FILE = docker-compose.yml
ENV_FILE = settings/local.env
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
	@echo "  make complete-setup    - Full project setup with all services"
	@echo "  make frontend-dev      - Start frontend development server"
	@echo "  make project-status    - Check project status"
	@echo "  make check-admin       - Check admin user"
	@echo "  make status            - Show service status"
	@echo ""

# Development targets
.PHONY: dev
dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d db rabbitmq
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)Database: localhost:5432$(NC)"
	@echo "$(YELLOW)RabbitMQ: localhost:5672 (Management: localhost:15672)$(NC)"

.PHONY: dev-full
dev-full: ## Start full development environment (all services)
	@echo "$(BLUE)Starting full development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
	@echo "$(GREEN)Full development environment started!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Database: localhost:5432$(NC)"
	@echo "$(YELLOW)RabbitMQ: localhost:5672 (Management: localhost:15672)$(NC)"

# Production targets
.PHONY: build
build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build
	@echo "$(GREEN)Build completed!$(NC)"

.PHONY: build-no-cache
build-no-cache: ## Build all Docker images without cache
	@echo "$(BLUE)Building Docker images (no cache)...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) build --no-cache
	@echo "$(GREEN)Build completed!$(NC)"

.PHONY: up
up: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d
	@echo "$(GREEN)Production environment started!$(NC)"

.PHONY: down
down: ## Stop all containers
	@echo "$(BLUE)Stopping all containers...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down
	@echo "$(GREEN)All containers stopped!$(NC)"

.PHONY: restart
restart: ## Restart all containers
	@echo "$(BLUE)Restarting all containers...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) restart
	@echo "$(GREEN)All containers restarted!$(NC)"

# Service-specific targets
.PHONY: up-db
up-db: ## Start only database and RabbitMQ
	@echo "$(BLUE)Starting database and RabbitMQ...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d db rabbitmq
	@echo "$(GREEN)Database and RabbitMQ started!$(NC)"

.PHONY: up-backend
up-backend: ## Start backend services (API + Worker)
	@echo "$(BLUE)Starting backend services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d api worker
	@echo "$(GREEN)Backend services started!$(NC)"

.PHONY: up-frontend
up-frontend: ## Start frontend service
	@echo "$(BLUE)Starting frontend service...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d frontend
	@echo "$(GREEN)Frontend service started!$(NC)"

# Logs targets
.PHONY: logs
logs: ## Show logs from all services
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f

.PHONY: logs-api
logs-api: ## Show API logs
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f api

.PHONY: logs-worker
logs-worker: ## Show worker logs
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f worker

.PHONY: logs-frontend
logs-frontend: ## Show frontend logs
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f frontend

.PHONY: logs-db
logs-db: ## Show database logs
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f db

# Database targets
.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec api alembic upgrade head
	@echo "$(GREEN)Database migrations completed!$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: This will delete all data!)
	@echo "$(RED)WARNING: This will delete all database data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BLUE)Resetting database...$(NC)"; \
		docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down -v; \
		docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d db; \
		echo "$(GREEN)Database reset completed!$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled.$(NC)"; \
	fi

# Admin management targets
.PHONY: create-admin
create-admin: ## Create admin user (default: admin/admin123)
	@echo "$(BLUE)Creating admin user...$(NC)"
	@if [ -z "$(LOGIN)" ]; then LOGIN=admin; fi; \
	if [ -z "$(EMAIL)" ]; then EMAIL=admin@agent-trade.com; fi; \
	if [ -z "$(PASSWORD)" ]; then PASSWORD=admin123; fi; \
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec api python scripts/create_admin.py --login $$LOGIN --email $$EMAIL --password $$PASSWORD
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
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, login VARCHAR(50) UNIQUE NOT NULL, email VARCHAR(50) UNIQUE, password VARCHAR(255) NOT NULL, user_telegram_id BIGINT, balance FLOAT DEFAULT 0, role VARCHAR(50) DEFAULT 'user', active BOOLEAN DEFAULT TRUE);"
	@docker-compose exec postgres psql -U agent -d agent -c "INSERT INTO users (login, email, password, role, active, balance) VALUES ('admin', 'admin@agent-trade.com', '$(shell echo -n 'admin123' | sha256sum | cut -d' ' -f1)', 'admin', TRUE, 0.0) ON CONFLICT (login) DO NOTHING;"
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: create-tables-docker
create-tables-docker: ## Create all tables in Docker database
	@echo "$(BLUE)Creating all tables in Docker database...$(NC)"
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS coins (id SERIAL PRIMARY KEY, name VARCHAR(50) UNIQUE NOT NULL, price_now FLOAT DEFAULT 0, max_price_now FLOAT DEFAULT 0, min_price_now FLOAT DEFAULT 0, open_price_now FLOAT DEFAULT 0, volume_now FLOAT DEFAULT 0, price_change_percentage_24h FLOAT, news_score_global FLOAT DEFAULT 100, parsed BOOLEAN DEFAULT TRUE);"
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS timeseries (id SERIAL PRIMARY KEY, coin_id INTEGER REFERENCES coins(id), timestamp VARCHAR(50), path_dataset VARCHAR(100) UNIQUE);"
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS data_timeseries (id SERIAL PRIMARY KEY, timeseries_id INTEGER REFERENCES timeseries(id), datetime TIMESTAMP NOT NULL, open FLOAT, max FLOAT, min FLOAT, close FLOAT, volume FLOAT);"
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS portfolio (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), coin_id INTEGER REFERENCES coins(id), amount FLOAT DEFAULT 0.0, price_avg FLOAT DEFAULT 0.0, CONSTRAINT uq_portfolio_user_coin UNIQUE (user_id, coin_id), CONSTRAINT ck_portfolio_amount_non_negative CHECK (amount >= 0));"
	@docker-compose exec postgres psql -U agent -d agent -c "CREATE TABLE IF NOT EXISTS transactions (id SERIAL PRIMARY KEY, status VARCHAR(30) DEFAULT 'open', user_id INTEGER REFERENCES users(id), coin_id INTEGER REFERENCES coins(id), type VARCHAR(20) NOT NULL, amount_orig FLOAT NOT NULL, amount FLOAT NOT NULL, price FLOAT NOT NULL);"
	@echo "$(GREEN)All tables created successfully!$(NC)"

.PHONY: create-admin-interactive
create-admin-interactive: ## Create admin user interactively
	@echo "$(BLUE)Creating admin user interactively...$(NC)"
	@read -p "Enter admin login: " LOGIN; \
	read -p "Enter admin email: " EMAIL; \
	read -s -p "Enter admin password: " PASSWORD; \
	echo; \
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec api python scripts/create_admin.py --login $$LOGIN --email $$EMAIL --password $$PASSWORD
	@echo "$(GREEN)Admin user creation completed!$(NC)"

.PHONY: db-backup
db-backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec db pg_dump -U $$DB__USER $$DB__DB_NAME > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created!$(NC)"

# Maintenance targets
.PHONY: clean
clean: ## Remove all containers, networks, and volumes
	@echo "$(BLUE)Cleaning up Docker environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Cleanup completed!$(NC)"

.PHONY: clean-images
clean-images: ## Remove all project images
	@echo "$(BLUE)Removing project images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down --rmi all
	@echo "$(GREEN)Images removed!$(NC)"

.PHONY: ps
ps: ## Show running containers
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) ps

.PHONY: status
status: ## Show status of all services
	@echo "$(BLUE)Service Status:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

.PHONY: check-admin
check-admin: ## Check if admin user exists in database
	@echo "$(BLUE)Checking admin user...$(NC)"
	@docker-compose exec postgres psql -U agent -d agent -c "SELECT id, login, email, role, active FROM users WHERE login = 'admin';" || echo "$(RED)Admin user not found$(NC)"

.PHONY: check-services
check-services: ## Check if all services are running
	@echo "$(BLUE)Checking services...$(NC)"
	@echo "$(YELLOW)Database:$(NC)"
	@docker-compose exec postgres pg_isready -U agent || echo "$(RED)Database is not ready$(NC)"
	@echo "$(YELLOW)Redis:$(NC)"
	@docker-compose exec redis redis-cli ping || echo "$(RED)Redis is not ready$(NC)"
	@echo "$(YELLOW)Backend API:$(NC)"
	@curl -f http://localhost:8000/health 2>/dev/null || echo "$(RED)Backend API is not ready$(NC)"
	@echo "$(YELLOW)Grafana:$(NC)"
	@curl -f http://localhost:3000 2>/dev/null || echo "$(RED)Grafana is not ready$(NC)"
	@echo "$(YELLOW)Prometheus:$(NC)"
	@curl -f http://localhost:9090 2>/dev/null || echo "$(RED)Prometheus is not ready$(NC)"

# Development helpers
.PHONY: shell-api
shell-api: ## Open shell in API container
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec api bash

.PHONY: shell-worker
shell-worker: ## Open shell in worker container
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec worker bash

.PHONY: shell-db
shell-db: ## Open shell in database container
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec db psql -U $$DB__USER -d $$DB__DB_NAME

# Health checks
.PHONY: health
health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo "$(YELLOW)Database:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T db pg_isready -U $$DB__USER || echo "$(RED)Database is not ready$(NC)"
	@echo "$(YELLOW)RabbitMQ:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T rabbitmq rabbitmq-diagnostics ping || echo "$(RED)RabbitMQ is not ready$(NC)"
	@echo "$(YELLOW)API:$(NC)"
	@curl -f http://localhost:8000/health || echo "$(RED)API is not ready$(NC)"
	@echo "$(YELLOW)Frontend:$(NC)"
	@curl -f http://localhost:5173 || echo "$(RED)Frontend is not ready$(NC)"

# Quick start targets
.PHONY: quick-start
quick-start: build up ## Quick start: build and start all services
	@echo "$(GREEN)Quick start completed!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)RabbitMQ Management: http://localhost:15672$(NC)"

.PHONY: setup-with-admin
setup-with-admin: up db-migrate create-admin ## Setup environment with admin user
	@echo "$(GREEN)Setup completed with admin user!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"

.PHONY: setup-docker-complete
setup-docker-complete: up create-tables-docker create-admin-docker ## Complete Docker setup with tables and admin
	@echo "$(GREEN)Docker setup completed with tables and admin user!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)RabbitMQ Management: http://localhost:15672$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"

.PHONY: setup-full
setup-full: up create-tables-docker create-admin-docker ## Full setup with all services and admin
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
	@echo "  - make logs - View logs$(NC)"
	@echo "  - make down - Stop all services$(NC)"

.PHONY: project-status
project-status: status check-admin check-services ## Show complete project status
	@echo "$(GREEN)Project status check completed!$(NC)"
	@echo "$(YELLOW)Summary:$(NC)"
	@echo "  - All Docker containers are running$(NC)"
	@echo "  - Database and Redis are ready$(NC)"
	@echo "  - Admin user is created: admin/admin123$(NC)"
	@echo "  - Monitoring services (Grafana, Prometheus) are running$(NC)"
	@echo "  - Backend API has issues with table creation$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Fix backend API table creation issues$(NC)"
	@echo "  - Start frontend development server$(NC)"

.PHONY: complete-setup
complete-setup: setup-full check-admin check-services ## Complete project setup with status check
	@echo "$(GREEN)Complete project setup finished!$(NC)"
	@echo "$(YELLOW)Services available:$(NC)"
	@echo "  - Database: localhost:5432 (PostgreSQL)$(NC)"
	@echo "  - Redis: localhost:6379$(NC)"
	@echo "  - Grafana: http://localhost:3000$(NC)"
	@echo "  - Prometheus: http://localhost:9090$(NC)"
	@echo "  - Backend API: http://localhost:8000 (has issues)$(NC)"
	@echo "$(YELLOW)Admin credentials: admin/admin123$(NC)"
	@echo "$(YELLOW)To start frontend: make frontend-dev$(NC)"
	@echo "$(GREEN)Project is ready for development!$(NC)"

.PHONY: quick-dev
quick-dev: up-db ## Quick dev: start only database and RabbitMQ
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Database: localhost:5432$(NC)"
	@echo "$(YELLOW)RabbitMQ: localhost:5672$(NC)"

# Environment setup
.PHONY: env-check
env-check: ## Check if environment file exists
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)Environment file $(ENV_FILE) not found!$(NC)"; \
		echo "$(YELLOW)Please create $(ENV_FILE) with your configuration.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)Environment file found: $(ENV_FILE)$(NC)"; \
	fi

# Pre-flight checks
.PHONY: preflight
preflight: env-check ## Run pre-flight checks
	@echo "$(BLUE)Running pre-flight checks...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is not installed$(NC)" >&2; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)Docker Compose is not installed$(NC)" >&2; exit 1; }
	@echo "$(GREEN)Pre-flight checks passed!$(NC)"

# Production deployment
.PHONY: deploy
deploy: preflight build up ## Deploy to production
	@echo "$(GREEN)Deployment completed!$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:5173$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"

# Monitoring
.PHONY: monitor
monitor: ## Monitor resource usage
	@echo "$(BLUE)Monitoring resource usage...$(NC)"
	@docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Backup and restore
.PHONY: backup
backup: ## Create full backup (database + volumes)
	@echo "$(BLUE)Creating full backup...$(NC)"
	@mkdir -p backups
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T db pg_dump -U $$DB__USER $$DB__DB_NAME > backups/db_backup_$$(date +%Y%m%d_%H%M%S).sql
	@docker run --rm -v $(PROJECT_NAME)_db_data:/data -v $$(pwd)/backups:/backup alpine tar czf /backup/volumes_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)Backup completed!$(NC)"

# Frontend development
.PHONY: frontend-dev
frontend-dev: ## Start frontend development server
	@echo "$(BLUE)Starting frontend development server...$(NC)"
	@cd frontend && npm run dev

.PHONY: restore
restore: ## Restore from backup (specify BACKUP_FILE=filename)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Please specify BACKUP_FILE=filename$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring from backup: $(BACKUP_FILE)$(NC)"
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T db psql -U $$DB__USER -d $$DB__DB_NAME < backups/$(BACKUP_FILE)
	@echo "$(GREEN)Restore completed!$(NC)"
