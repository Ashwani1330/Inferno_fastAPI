# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Welcome to the Inferno VR Fire-Safety Training API"}

@router.get("/analytics")
async def redirect_to_dashboard():
    """Redirects to the analytics dashboard."""
    return RedirectResponse(url="/api/dashboard")