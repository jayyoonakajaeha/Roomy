from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 로컬 환경 변수 로드
load_dotenv()

from app.matching.router import router as matching_router
from app.repair.router import router as repair_router
from app.users.router import router as users_router

app = FastAPI(
    title="Roommate Matching & Facility Repair API",
    description="Combined API for roommate matching and facility repair AI services.",
    version="0.1.0"
)

# CORS Middleware (React 등 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 구체적인 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(matching_router, prefix="/api/matching", tags=["Matching"])
app.include_router(repair_router, prefix="/api/repair", tags=["Repair"])
app.include_router(users_router, prefix="/api/users", tags=["Users"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Roommate Matching & Repair API"}

# Forced reload trigger
