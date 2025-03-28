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
        
        # Ensure scatterplot_matrix is generated
        if 'plots' in dashboard_data and 'scatterplot_matrix' not in dashboard_data['plots']:
            logger.info("Generating missing scatterplot matrix visualization")
            performances = await mongo_service.get_performances_optimized()
            if performances:
                import pandas as pd
                df = pd.DataFrame(performances)
                scatterplot_matrix = await analytics_service.generate_scatterplot_matrix(df)
                if scatterplot_matrix:
                    dashboard_data['plots']['scatterplot_matrix'] = scatterplot_matrix
                    await analytics_service.update_dashboard_data(dashboard_data)
                    logger.info("Scatterplot matrix generated successfully")
                else:
                    logger.error("Failed to generate scatterplot matrix")
        
        # Add error handling for template issues
        try:
            # Render template with data
            return templates.TemplateResponse(
                "dashboard.html", 
                {
                    "request": request,
                    "stats": dashboard_data.get("stats", {}),
                    "plots": dashboard_data.get("plots", {}),
                    "latest_records": dashboard_data.get("latest_records", [])
                }
            )
        except Exception as template_error:
            logger.error(f"Template rendering error: {str(template_error)}")
            # Return a more detailed error page with template debugging info
            error_html = f"""
            <h1>Dashboard Template Error</h1>
            <p>There was an error rendering the dashboard template.</p>
            <h2>Error details:</h2>
            <pre>{str(template_error)}</pre>
            <h2>Available template data:</h2>
            <ul>
                <li>Stats keys: {', '.join(dashboard_data.get('stats', {}).keys()) or 'None'}</li>
                <li>Plots keys: {', '.join(dashboard_data.get('plots', {}).keys()) or 'None'}</li>
                <li>Latest records count: {len(dashboard_data.get('latest_records', []))}</li>
            </ul>
            """
            return HTMLResponse(error_html)
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return HTMLResponse(f"<h1>Error generating dashboard</h1><p>{str(e)}</p>")

@router.get("/dashboard/refresh")
async def refresh_dashboard():
    """Force regeneration of dashboard analytics"""
    try:
        # Process data
        success = await analytics_service.process_data()
        
        if success:
            return {"status": "success", "message": "Dashboard data refreshed successfully"}
        else:
            return {"status": "error", "message": "Failed to refresh dashboard data"}
    except Exception as e:
        logger.error(f"Error refreshing dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
