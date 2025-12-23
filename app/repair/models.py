from pydantic import BaseModel
from typing import List, Optional

class RepairAnalysisResult(BaseModel):
    title: str # 게시물 제목 (한글, 간결한 명사형)
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
    analysis: Optional[RepairAnalysisResult] = None  # 중복이면 None
    duplicates: List[DuplicateReportInfo]
    is_new: bool
    newReportId: Optional[int] = None  # 중복이 아닌 경우에만 할당된 새 ID

class RepairRequest(BaseModel):
    existingReportIds: List[int] = []  # 백엔드에서 위치 필터링한 기존 게시물 ID 목록
    totalReportCount: int  # 현재 총 게시물 수 (새 ID = totalReportCount + 1)
    floor: str  # 층수
    room_number: Optional[str] = None  # 호수 (공용시설이면 null)
