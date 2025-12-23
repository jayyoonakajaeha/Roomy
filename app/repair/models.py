from typing import List, Optional
from pydantic import BaseModel
from typing import List, Optional

class RepairAnalysisResult(BaseModel):
    """Gemini Analysis Result"""
    item: str      # ex: toilet, sink, door
    issue: str     # ex: clogged, broken_hinge
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    priority_score: int # 1~10
    reasoning: str # 우선순위 판단 근거
    repair_suggestion: str
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
    imagePath: str
    vectorPath: str
    building: str
    floor: str
    room_number: Optional[str] = None
