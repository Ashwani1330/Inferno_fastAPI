import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "test")

# Email settings
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.example.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@inferno-vr.com")

# API settings
API_TITLE = "Inferno VR Fire-Safety Training API"
API_DESCRIPTION = "API for Immersive Navigation for Fire Emergency Response & Neutralization Operations research"
API_VERSION = "1.0.0"