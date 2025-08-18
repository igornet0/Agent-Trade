import uvicorn
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core import settings
from core.utils import setup_logging
from backend.app import create_app

main_app = create_app(create_custom_static_urls=True)

if __name__ == "__main__":
    setup_logging()
    
    print(f"Starting server on {settings.run.host}:{settings.run.port}")
    
    uvicorn.run(
        "run_app_simple:main_app",
        host=settings.run.host,
        port=settings.run.port,
        workers=1,
        reload=False,
        log_level="info"
    )
