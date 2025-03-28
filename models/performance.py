from pydantic import BaseModel
from typing import Optional

# Performance data models for the Inferno VR application
# These models handle input from VR sessions and output responses

class PerformanceInput(BaseModel):
    email: Optional[str] = None
    age: str
    sceneType: str
    difficulty: str
    timeToFindExtinguisher: float
    timeToExtinguishFire: float
    timeToTriggerAlarm: float
    timeToFindExit: float

class PerformanceOutput(BaseModel):
    message: str
    performanceScore: float
