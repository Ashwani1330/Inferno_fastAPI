from pydantic import BaseModel
from typing import Optional

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
