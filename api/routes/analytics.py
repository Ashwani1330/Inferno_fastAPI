from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import os
import logging
from fastapi_cache.decorator import cache

from services.mongo_service import MongoService
from services.analytics_service import AnalyticsService
from tasks.background_tasks import trigger_analytics_update

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
mongo_service = MongoService()
analytics_service = AnalyticsService()

# Set up templates
templates_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates")
templates = Jinja2Templates(directory=templates_path)

@router.get("/dashboard", response_class=HTMLResponse)
# @cache(expire=60 * 5)  # Cache for 5 minutes
async def analytics_dashboard(request: Request):
    """Serves a comprehensive analytics dashboard for research purposes."""
    try:
        # Get pre-processed dashboard data
        dashboard_data = await analytics_service.get_dashboard_data()
        
        if not dashboard_data:
            # If no data available, process on-demand
            success = await analytics_service.process_data()
            if not success:
                return HTMLResponse("<h1>No data available yet</h1>")
            dashboard_data = await analytics_service.get_dashboard_data()
        
        # Render template with data
        return templates.TemplateResponse(
            "dashboard.html", 
            {
                "request": request,
                "stats": dashboard_data["stats"],
                "plots": dashboard_data["plots"],
                "latest_records": dashboard_data["latest_records"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return HTMLResponse(f"<h1>Error generating dashboard</h1><p>{str(e)}</p>")

@router.post("/notify-new-data")
async def notify_new_data():
    """Endpoint to notify when new performance data is added"""
    try:
        # Trigger background analytics update
        await trigger_analytics_update()
        return {"status": "success", "message": "Analytics update triggered"}
    except Exception as e:
        logger.error(f"Error triggering analytics update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
