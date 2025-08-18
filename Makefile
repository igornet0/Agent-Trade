# Makefile for Agent Trade Project
# Usage: make <target>

# Variables
PROJECT_NAME = agent-trade
COMPOSE_FILE = docker-compose.yml
ENV_FILE = settings/prod.env
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

.PHONY: restore
restore: ## Restore from backup (specify BACKUP_FILE=filename)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Please specify BACKUP_FILE=filename$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring from backup: $(BACKUP_FILE)$(NC)"
	@docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) exec -T db psql -U $$DB__USER -d $$DB__DB_NAME < backups/$(BACKUP_FILE)
	@echo "$(GREEN)Restore completed!$(NC)"
