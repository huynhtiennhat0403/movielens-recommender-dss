from fastapi import APIRouter
from backend.dependencies import get_model_service, get_tmdb_service
from backend.schemas import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def health():
    model = get_model_service()
    tmdb = get_tmdb_service()
    return {
        "status": "ok",
        "device": str(model.device),
        "model": "NCF",
        "tmdb_enabled": tmdb.enabled,
    }
