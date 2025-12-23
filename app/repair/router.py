from typing import Optional
from fastapi import APIRouter
from .service import process_repair_request
from .models import RepairResponse, RepairRequest

router = APIRouter()

@router.post("/analyze", response_model=RepairResponse, summary="Analyze repair item & Check duplicates")
async def analyze_repair(request: RepairRequest):
    """
    **시설물 고장 신고 분석 API**
    
    - **입력**: JSON (이미지 경로, 벡터 경로, 건물, 층, 호수)
    - **기능**:
      1. **AI 분석 (Gemini)**: 고장 항목, 심각도(Priority), 수리 제안(한글) 추출.
      2. **중복 감지 (CLIP)**: 동일 위치 & 벡터 유사도 기반 중복 확인.
    - **출력**: 분석 결과 및 중복 의심 리스트.
    """
    result = await process_repair_request(request)
    return result
