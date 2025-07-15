from fastapi import APIRouter
from .health import get_health_status

router = APIRouter()


@router.get("/health")
async def health_check():
    return await get_health_status()
