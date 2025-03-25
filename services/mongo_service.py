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
