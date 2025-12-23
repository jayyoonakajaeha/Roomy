from fastapi import APIRouter
from typing import List
from .models import MatchRequest, MatchResult
from .service import calculate_hybrid_match

router = APIRouter()

@router.post("/match", response_model=List[MatchResult], summary="Get roommate matches")
async def match_roommates(request: MatchRequest):
    """
    룸메이트 매칭 엔드포인트
    """
    if not request.candidates:
        return []
    matches = calculate_hybrid_match(request)
    return matches
