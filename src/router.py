from fastapi import APIRouter
import src.endpoint as endpoint

router = APIRouter()

router.include_router(endpoint.router, prefix="/agent", tags=["agent"])
