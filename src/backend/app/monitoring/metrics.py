from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

# Common, reusable metrics registry objects

request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_latency_seconds = Histogram(
    'http_request_latency_seconds',
    'Latency of HTTP requests in seconds',
    ['endpoint', 'method']
)

# Celery tasks metrics
celery_tasks_total = Counter(
    'celery_tasks_total',
    'Total Celery tasks executed',
    ['task_name', 'status']
)

celery_task_duration_seconds = Histogram(
    'celery_task_duration_seconds',
    'Duration of Celery tasks in seconds',
    ['task_name']
)

# ML specific metrics
ml_models_trained_total = Counter(
    'ml_models_trained_total',
    'Number of ML models trained',
    ['agent_type']
)

ml_models_tested_total = Counter(
    'ml_models_tested_total',
    'Number of ML models tested',
    ['agent_type']
)


def export_prometheus() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


