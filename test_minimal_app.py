import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = FastAPI(title="ML Trading System - Test", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-trading-system"}

@app.get("/metrics")
async def metrics():
    """Простой endpoint для метрик Prometheus"""
    return """# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health"} 1
http_requests_total{method="GET",endpoint="/metrics"} 1

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 2
http_request_duration_seconds_bucket{le="0.5"} 2
http_request_duration_seconds_bucket{le="1.0"} 2
http_request_duration_seconds_bucket{le="+Inf"} 2
http_request_duration_seconds_sum 0.05
http_request_duration_seconds_count 2

# HELP ml_model_train_total Total number of ML model training tasks
# TYPE ml_model_train_total counter
ml_model_train_total{model_type="AgentNews"} 0
ml_model_train_total{model_type="AgentPredTime"} 0
ml_model_train_total{model_type="AgentTradeTime"} 0
ml_model_train_total{model_type="AgentRisk"} 0
ml_model_train_total{model_type="AgentTradeAggregator"} 0

# HELP ml_train_duration_seconds ML training duration
# TYPE ml_train_duration_seconds histogram
ml_train_duration_seconds_bucket{le="60"} 0
ml_train_duration_seconds_bucket{le="300"} 0
ml_train_duration_seconds_bucket{le="600"} 0
ml_train_duration_seconds_bucket{le="+Inf"} 0
ml_train_duration_seconds_sum 0
ml_train_duration_seconds_count 0
"""

@app.get("/api_db_agent/models")
async def list_models():
    """Список моделей для тестирования"""
    return {
        "models": [
            {"id": 1, "name": "News Model 1", "type": "AgentNews", "status": "SUCCESS", "timeframe": "5m", "version": "1.0"},
            {"id": 2, "name": "Pred Time Model 1", "type": "AgentPredTime", "status": "SUCCESS", "timeframe": "5m", "version": "1.0"},
            {"id": 3, "name": "Trade Time Model 1", "type": "AgentTradeTime", "status": "PROGRESS", "timeframe": "5m", "version": "1.0"},
            {"id": 4, "name": "Risk Model 1", "type": "AgentRisk", "status": "SUCCESS", "timeframe": "5m", "version": "1.0"},
            {"id": 5, "name": "Trade Aggregator 1", "type": "AgentTradeAggregator", "status": "PENDING", "timeframe": "5m", "version": "1.0"},
        ]
    }

@app.post("/api_db_agent/test_model")
async def test_model(request: dict):
    """Тестовый endpoint для тестирования моделей"""
    return {"task_id": "test-task-123", "status": "accepted"}

@app.get("/api_db_agent/task/{task_id}")
async def get_task_status(task_id: str):
    """Статус задачи"""
    return {
        "task_id": task_id,
        "state": "SUCCESS",
        "status": "Task completed successfully",
        "meta": {
            "test_results": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "charts": [
                    {
                        "title": "Prediction vs Actual",
                        "type": "line",
                        "data": {
                            "actual": [100, 101, 102, 103, 104],
                            "predicted": [99, 101, 102, 104, 105]
                        }
                    }
                ],
                "recommendations": [
                    "Model performance is good",
                    "Consider retraining with more data",
                    "Monitor for concept drift"
                ]
            }
        }
    }

if __name__ == "__main__":
    print("Starting minimal test server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
