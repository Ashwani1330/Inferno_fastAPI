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