from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from .service import process_repair_request
from .models import RepairResponse

router = APIRouter()

@router.post("/analyze", response_model=RepairResponse, summary="Analyze repair item & Check duplicates")
async def analyze_repair(
    file: UploadFile = File(...),
    building: str = Form(..., description="Building Name (e.g., A-Dorm)"),
    floor: str = Form(..., description="Floor Number (e.g., 3)"),
    room_number: Optional[str] = Form(None, description="Room Number (e.g., 1201) - Optional for private rooms")
):
    """
    **시설물 고장 신고 분석 API**
    
    - **입력**: 파손된 사진 파일, 건물명, 층수, (선택) 호수.
    - **기능**:
      1. **AI 분석 (Gemini)**: 고장 항목, 심각도(Priority), 수리 제안(한글) 추출.
      2. **중복 감지 (CLIP)**: 동일 위치(건물/층/호수)에 접수된 유사한 사진이 있는지 확인.
    - **출력**: 분석 결과 및 중복 의심 리스트.
    """
    result = await process_repair_request(file, building, floor, room_number)
    return result
