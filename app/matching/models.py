from typing import List, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

# ==========================================
# 📐 Enums & Constants
# ==========================================

class CleaningCycle(str, Enum):
    DAILY = "DAILY"
    EVERY_TWO_DAYS = "EVERY_TWO_DAYS"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    NEVER = "NEVER"

    def to_score(self) -> int:
        mapping = {
            "DAILY": 0,
            "EVERY_TWO_DAYS": 1,
            "WEEKLY": 2,
            "MONTHLY": 3,
            "NEVER": 4
        }
        return mapping[self.value]

class DrinkingStyle(str, Enum):
    RARELY = "RARELY"
    SOMETIMES = "SOMETIMES"
    FREQUENTLY = "FREQUENTLY"
    
    def to_score(self) -> int:
        mapping = {
            "RARELY": 0,
            "SOMETIMES": 1,
            "FREQUENTLY": 2
        }
        return mapping[self.value]

# ==========================================
# 📐 Pydantic Models
# ==========================================

class UserProfile(BaseModel):
    """후보자(Candidate) 및 내 정보(My Profile) 모델 - DB Schema 일치"""
    id: int 
    gender: str # 'MALE', 'FEMALE'
    name: str
    birthYear: int
    kakaoId: Optional[str] = None
    mbti: Optional[str] = None
    
    # Flags
    smoker: bool
    snoring: bool
    bugKiller: bool
    
    # Time (Scale Input)
    sleepTime: int 
    wakeTime: int
    
    # Enums
    cleaningCycle: CleaningCycle
    drinkingStyle: DrinkingStyle
    
    absentDays: Optional[List[str]] = []
    hobby: Optional[str] = None
    
    # Text Descriptions
    selfDescription: Optional[str] = None
    roommateDescription: Optional[str] = None
    
    # Embeddings
    # selfIntroductionEmbedding: Vector of selfDescription (Candidate uses this)
    selfIntroductionEmbedding: Optional[List[float]] = None
    # roommateCriteriaEmbedding: Vector of roommateDescription (Seeker uses this)
    roommateCriteriaEmbedding: Optional[List[float]] = None

    @property
    def age(self) -> int:
        return datetime.now().year - self.birthYear


class UserPreferences(BaseModel):
    """검색 조건"""
    targetGender: str
    targetAgeRange: Tuple[int, int] # (min, max)
    
    # 중요 체크리스트 (가산점 항목)
    preferNonSmoker: bool = False
    preferGoodAtBugs: bool = False
    preferQuietSleeper: bool = False # 코골이 X
    preferNonDrinker: bool = False
    
    # Note: Text queries are now handled via UserProfile.roommateDescription embedding
    
    
class MatchRequest(BaseModel):
    myProfile: UserProfile
    preferences: UserPreferences
    candidates: List[UserProfile]


class MatchResult(BaseModel):
    userId: int
    name: str
    totalScore: float
    rank: int
    matchDetails: dict
