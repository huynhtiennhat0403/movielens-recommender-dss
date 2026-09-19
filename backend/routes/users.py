from fastapi import APIRouter, HTTPException, Query
from backend.dependencies import get_model_service, get_profile_store, get_recommender_service, get_user_service
from backend.schemas import ExistingUserSummary, HistoryResponse, RatingCreateRequest, RecommendationResponse

router = APIRouter(prefix="/users", tags=["Existing users"])

@router.get("/{user_id}", response_model=ExistingUserSummary)
def get_user(user_id: int):
    try:
        return get_user_service().existing_user_summary(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@router.get("/{user_id}/history", response_model=HistoryResponse)
def history(user_id: int):
    try:
        return {
            "subject_id": str(user_id),
            "subject_type": "existing_user",
            "ratings": get_user_service().existing_user_history(user_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@router.get("/{user_id}/recommendations", response_model=RecommendationResponse)
def recommendations(user_id: int, top_k: int = Query(default=10, ge=1, le=50)):
    try:
        user_service = get_user_service()
        base_history = user_service.raw_ratings[
            user_service.raw_ratings["user_id"] == int(user_id)
        ]["movie_id"].astype(int).tolist()
        return get_recommender_service().existing_user_recommendations(
            user_id, base_history, top_k
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/{user_id}/ratings")
def add_rating(user_id: int, payload: RatingCreateRequest):
    model = get_model_service()
    try:
        model.validate_user_id(user_id)
        model.validate_movie_id(payload.movie_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    get_profile_store().upsert_existing_user_rating(user_id, payload.movie_id, payload.rating)
    return {
        "status": "saved",
        "user_id": user_id,
        "movie_id": payload.movie_id,
        "rating": payload.rating,
    }
