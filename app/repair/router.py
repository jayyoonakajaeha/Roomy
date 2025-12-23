from fastapi import APIRouter, UploadFile, File
from .service import analyze_repair_image

router = APIRouter()

@router.post("/analyze", summary="Analyze facility repair image")
async def analyze_repair(file: UploadFile = File(...)):
    """
    고장 사진을 업로드하면 AI가 분석하여 수리 필요한 항목, 카테고리, 심각도 등을 반환합니다.
    """
    result = await analyze_repair_image(file)
    return result
