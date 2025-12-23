from fastapi import FastAPI
from dotenv import load_dotenv

# 로컬 환경 변수 로드
load_dotenv()

from .matching.router import router as matching_router
from .repair.router import router as repair_router

app = FastAPI(title="Roommate Matching & Facility Repair API", description="Combined API for roommate matching and facility repair AI services.")

# Register Routers
app.include_router(matching_router, prefix="/api/matching", tags=["Matching"])
app.include_router(repair_router, prefix="/api/repair", tags=["Repair"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Roommate Matching & Repair API"}
