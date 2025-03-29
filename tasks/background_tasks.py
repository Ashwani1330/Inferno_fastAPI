import asyncio
import logging
from datetime import datetime
import time
from functools import wraps

from services.analytics_service import AnalyticsService
from services.mongo_service import MongoService

logger = logging.getLogger(__name__)

# Task registry
_tasks = {}
_last_run = {}

analytics_service = AnalyticsService()
mongo_service = MongoService()

def background_task(interval_minutes=60):
    """Decorator to register a background task"""
    def decorator(func):
        task_name = func.__name__
        _tasks[task_name] = {
            'func': func,
            'interval': interval_minutes * 60,  # Convert to seconds
            'is_running': False
        }
        _last_run[task_name] = 0  # Never run
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

@background_task(interval_minutes=60)  # Run every hour
async def update_analytics():
    """Update analytics data periodically"""
    try:
        logger.info("Running scheduled analytics update")
        await analytics_service.process_data()
        logger.info("Completed scheduled analytics update")
    except Exception as e:
        logger.error(f"Error in scheduled analytics update: {str(e)}")

@background_task(interval_minutes=5)  # Check for new data every 5 minutes
async def check_new_data():
    """Check for new data and trigger analytics update if needed"""
    try:
        should_update = await analytics_service.should_regenerate()
        if should_update:
            logger.info("New data detected, triggering analytics update")
            await analytics_service.process_data()
    except Exception as e:
        logger.error(f"Error checking for new data: {str(e)}")

async def run_task(task_name):
    """Run a specific task immediately"""
    if task_name in _tasks:
        task = _tasks[task_name]
        if not task['is_running']:
            task['is_running'] = True
            try:
                await task['func']()
                _last_run[task_name] = time.time()
            except Exception as e:
                logger.error(f"Error running task {task_name}: {str(e)}")
            finally:
                task['is_running'] = False
        else:
            logger.warning(f"Task {task_name} is already running")
    else:
        logger.error(f"Task {task_name} not found")

async def task_scheduler():
    """Main scheduler loop to run background tasks"""
    logger.info("Starting background task scheduler")
    
    # Run analytics update immediately on startup
    await run_task('update_analytics')
    
    while True:
        try:
            current_time = time.time()
            
            for task_name, task in _tasks.items():
                # Check if it's time to run the task
                if (current_time - _last_run.get(task_name, 0)) >= task['interval']:
                    # Run the task if it's not already running
                    if not task['is_running']:
                        # Run task in background without waiting
                        asyncio.create_task(run_task(task_name))
            
            # Sleep briefly to avoid hogging CPU
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in task scheduler: {str(e)}")
            await asyncio.sleep(60)  # Sleep longer if there's an error

def start_background_tasks():
    """Start the background task scheduler"""
    asyncio.create_task(task_scheduler())
    logger.info("Background tasks scheduler started")

# Function to trigger analytics update when new performance is submitted
async def trigger_analytics_update():
    """Trigger analytics update when new data is submitted"""
    # Trigger analytics update that now includes additional visualization processing
    await analytics_service.process_data()
