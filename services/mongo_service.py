import motor.motor_asyncio
from dotenv import load_dotenv
import os

load_dotenv()

class MongoService:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGO_URI"))
        self.db = self.client["test"]
        self.performance_collection = self.db["performances"]

    async def insert_performance(self, data):
        await self.performance_collection.insert_one(data)

    async def get_performances(self):
        return await self.performance_collection.find().to_list(1000)

    async def get_performances_optimized(self):
        # Select only the fields you need
        projection = {
            "email": 1, 
            "age": 1, 
            "sceneType": 1, 
            "difficulty": 1, 
            "timeToFindExtinguisher": 1,
            "timeToExtinguishFire": 1, 
            "timeToTriggerAlarm": 1, 
            "timeToFindExit": 1,
            "performanceScore": 1, 
            "timestamp": 1
        }

        return await self.performance_collection.find({}, projection).to_list(1000)
