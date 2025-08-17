# Agent-Trade

Quick start (Docker)

1. Copy settings/prod.env and adjust DB/Celery URLs
2. Build and run:
   - make build
   - make up

Services
- API: FastAPI (uvicorn)
- Worker: Celery
- Frontend: Vite React
- DB: Postgres
