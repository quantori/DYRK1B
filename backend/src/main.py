import logging

from fastapi import FastAPI, Request, Response
from motor.motor_asyncio import AsyncIOMotorClient

from uvicorn import run
from apps.authorization.routers import router as authorization_router
from apps.engine.routers import router as engine_router
from apps.entity.routers import router as entity_router

from config import settings

log = logging.getLogger(__name__)

app = FastAPI()
app.soft_limit_counter = {}  # type: ignore
app.hard_limit_counter = {}  # type: ignore


@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(settings.DB_URL)
    app.db = app.mongodb_client[settings.DB_NAME]


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

app.include_router(entity_router, tags=["entity"], prefix="/entity")
app.include_router(authorization_router, tags=["authorization"], prefix="/authorization")
app.include_router(engine_router, tags=["engine"], prefix="/engine")

if __name__ == "__main__":
    run("main:app",
        host=settings.HOST,
        reload=settings.DEBUG_MODE,
        port=settings.PORT,
        log_config=settings.LOG_CONFIG,
        )