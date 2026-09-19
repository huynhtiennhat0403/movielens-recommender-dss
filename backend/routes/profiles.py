from fastapi import APIRouter, HTTPException, Query
from backend.dependencies import get_model_service, get_profile_store, get_recommender_service, get_user_service
from backend.schemas import HistoryResponse, NewProfileCreateRequest, NewProfileResponse, RatingCreateRequest, RecommendationResponse

router = APIRouter(prefix="/profiles", tags=["New users"])

@router.post("", response_model=NewProfileResponse)
def create_profile(payload: NewProfileCreateRequest):
    model = get_model_service()
    try:
        for item in payload.ratings:
            model.validate_movie_id(item.movie_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    profile_id = get_profile_store().create_profile(
        [(x.movie_id, x.rating) for x in payload.ratings]
    )
    return {"profile_id": profile_id, "rating_count": len(payload.ratings)}

@router.get("/{profile_id}/history", response_model=HistoryResponse)
def history(profile_id: str):
    try:
        return {
            "subject_id": profile_id,
            "subject_type": "new_profile",
            "ratings": get_user_service().new_profile_history(profile_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@router.get("/{profile_id}/recommendations", response_model=RecommendationResponse)
def recommendations(profile_id: str, top_k: int = Query(default=10, ge=1, le=50)):
    try:
        return get_recommender_service().new_profile_recommendations(profile_id, top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/{profile_id}/ratings")
def add_rating(profile_id: str, payload: RatingCreateRequest):
    try:
        get_model_service().validate_movie_id(payload.movie_id)
        get_profile_store().upsert_new_profile_rating(profile_id, payload.movie_id, payload.rating)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "status": "saved",
        "profile_id": profile_id,
        "movie_id": payload.movie_id,
        "rating": payload.rating,
    }
