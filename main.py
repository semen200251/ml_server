from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from src.routers import v1_router

def create_application():
    return FastAPI(
        title="Ml_Service",
        description="Ml сервис для проекта",
        version="0.0.1",
        responses={404: {"description": "Not Found!"}},
    )


app = create_application()


def _configure():
    app.include_router(v1_router)


_configure()