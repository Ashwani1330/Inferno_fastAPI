from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this import
from fastapi.templating import Jinja2Templates
import os
from tasks.background_tasks import start_background_tasks

# Import all routers
from api.routes.base import router as base_router
from api.routes.performance import router as performance_router
from api.routes.analytics import router as analytics_router

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
import os

# Create FastAPI app
app = FastAPI(
    title="Inferno VR Fire-Safety Training API",
    description="API for Immersive Navigation for Fire Emergency Response & Neutralization Operations research",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include all routers
app.include_router(base_router)
app.include_router(performance_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    FastAPICache.init(RedisBackend(redis), prefix="inferno-dashboard-cache")

# Start background tasks on application startup
@app.on_event("startup")
async def startup_event():
    # ...existing startup code...
    
    # Start background tasks
    start_background_tasks()