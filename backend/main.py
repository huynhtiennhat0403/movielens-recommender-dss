from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.health import router as health_router
from backend.routes.movies import router as movies_router
from backend.routes.users import router as users_router
from backend.routes.profiles import router as profiles_router

app = FastAPI(
    title="MovieLens Recommender DSS API",
    version="0.2.0",
    description="NCF recommender backend with TMDB metadata enrichment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(movies_router)
app.include_router(users_router)
app.include_router(profiles_router)

@app.get("/")
def root():
    return {"name": "MovieLens Recommender DSS API", "docs": "/docs"}


@app.get("/runtime")
def runtime():
    return {
        "api_version": "0.3.0",
        "frontend_origins": [
            "http://localhost:3000",
            "http://localhost:5173"
        ]
    }
