from fastapi import APIRouter

from .predict import predict_router
from .preprocessing_service import preprocessing_router

v1_router = APIRouter(
    tags=["v1"],
    prefix=""
)

v1_router.include_router(predict_router)
v1_router.include_router(preprocessing_router)