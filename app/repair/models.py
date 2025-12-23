from pydantic import BaseModel
from typing import List, Optional

class RepairAnalysisResult(BaseModel):
    item: str  # 고장 물건
    issue: str # 문제 현상
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    priority_score: int # 1~10
    reasoning: str # 우선순위 판단 근거
    description: str # 한글 설명

class DuplicateReportInfo(BaseModel):
    reportId: int
    similarity: float
    description: str
    location: str
    image_url: Optional[str] = None

class RepairResponse(BaseModel):
    analysis: RepairAnalysisResult
    duplicates: List[DuplicateReportInfo]
    is_new: bool

class RepairRequest(BaseModel):
    imagePath: str  # 신규 신고 이미지 경로 (CLIP 임베딩 계산용)
    existingReportIds: List[int] = []  # 백엔드에서 위치 필터링한 기존 게시물 ID 목록
    floor: str  # 층수
    room_number: Optional[str] = None  # 호수 (공용시설이면 null)
