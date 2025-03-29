from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import os
import logging
from fastapi_cache.decorator import cache
import pandas as pd
import io
import re
import hashlib
import matplotlib

# Ensure matplotlib is configured correctly
matplotlib.use('Agg')  # Force non-interactive backend

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

# Email anonymization function
def anonymize_email(text):
    """
    Anonymize email addresses found in text.
    Preserves domain but replaces username with a hash.
    """
    if not isinstance(text, str):
        return text
        
    # Regex to find email addresses
    email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    
    # Find all email addresses in the text
    matches = re.findall(email_pattern, text)
    
    # If no emails found, return original text
    if not matches:
        return text
        
    # Replace each email with anonymized version
    anonymized_text = text
    for username, domain in matches:
        original_email = f"{username}@{domain}"
        # Create a hash of the username (take first 8 chars of hash)
        hashed_username = hashlib.md5(username.encode()).hexdigest()[:8]
        anonymized_email = f"{hashed_username}@{domain}"
        anonymized_text = anonymized_text.replace(original_email, anonymized_email)
    
    return anonymized_text

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
        
        # Update alternative visualization keys
        alt_viz_keys = ['correlation_heatmap', 'scatterplot_matrix']
        
        # If alternative visualizations aren't in dashboard data, generate them
        missing_viz = all(viz not in dashboard_data.get('plots', {}) for viz in alt_viz_keys)
        
        if missing_viz:
            logger.info("Generating missing alternative visualizations")
            try:
                # Set a timeout for the visualization generation (30 seconds)
                import asyncio
                performances = await mongo_service.get_performances_optimized()
                if performances and len(performances) >= 10:
                    import pandas as pd
                    df = pd.DataFrame(performances)
                    
                    # Generate the alternative visualizations with timeout
                    try:
                        # Run in executor to avoid matplotlib threading issues
                        loop = asyncio.get_event_loop()
                        alt_viz = await asyncio.wait_for(
                            loop.run_in_executor(None, 
                                lambda: analytics_service.generate_alternative_visualizations(df)),
                            timeout=30.0
                        )
                        
                        if alt_viz:
                            # Add to dashboard data
                            for viz_key, viz_data in alt_viz.items():
                                if viz_data and 'base64' in viz_data:
                                    dashboard_data['plots'][viz_key] = viz_data
                            
                            await analytics_service.update_dashboard_data(dashboard_data)
                            logger.info(f"Alternative visualizations generated successfully")
                    except asyncio.TimeoutError:
                        logger.error("Timeout occurred while generating alternative visualizations")
                else:
                    logger.warning(f"Insufficient data for alternative visualizations: {len(performances) if performances else 0} records")
            except Exception as e:
                logger.error(f"Error during alternative visualization generation: {str(e)}")
        
        # Always create a valid plots entry even if empty
        if 'plots' not in dashboard_data:
            dashboard_data['plots'] = {}
        
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

@router.get("/export/{export_type}")
async def export_data(export_type: str):
    """
    Export performance data as CSV or Excel.
    URL example: /api/export/csv or /api/export/excel
    """
    try:
        performances = await mongo_service.get_performances_optimized()
        if not performances:
            raise HTTPException(status_code=404, detail="No performance data available")
        df = pd.DataFrame(performances)
        
        # Remove manual anonymization and use the utility function for comprehensive anonymization
        from utils.anonymization import anonymize_dataframe
        df = anonymize_dataframe(df)
        
        # Log columns after anonymization for debugging
        logger.info(f"Columns after anonymization: {df.columns.tolist()}")
        
        # Final verification to ensure no emails are in the data
        stream = io.BytesIO()
        if export_type.lower() == "csv":
            df.to_csv(stream, index=False)
            media_type = "text/csv"
            filename = "anonymized_performance_data.csv"
        elif export_type.lower() == "excel":
            try:
                # Try using openpyxl
                with pd.ExcelWriter(stream, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
            except ModuleNotFoundError:
                try:
                    # Fallback to xlsxwriter
                    with pd.ExcelWriter(stream, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False)
                except ModuleNotFoundError:
                    error_msg = (
                        "Excel export libraries are not installed. Please install required packages using:\n"
                        "pip install openpyxl xlsxwriter\n\n"
                        "Alternatively, you can use CSV export which doesn't require additional packages."
                    )
                    raise HTTPException(status_code=500, detail=error_msg)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "anonymized_performance_data.xlsx"
        else:
            raise HTTPException(status_code=400, detail="Invalid export type")
        
        logger.info(f"Successfully exported anonymized data as {export_type}")
        stream.seek(0)
        return StreamingResponse(
            stream, 
            media_type=media_type, 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
