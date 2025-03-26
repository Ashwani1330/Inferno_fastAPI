import motor.motor_asyncio
from core.config import MONGO_URI, MONGO_DB

class MongoDB:
    client = None
    db = None
    
    @classmethod
    def get_client(cls):
        if cls.client is None:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        return cls.client
    
    @classmethod
    def get_db(cls):
        if cls.db is None:
            cls.db = cls.get_client()[MONGO_DB]
        return cls.db