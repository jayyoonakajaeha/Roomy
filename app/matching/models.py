from typing import List, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# ==========================================
# 📐 Enums & Constants
# ==========================================

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"

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
    gender: Gender
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
    
    
    # Embeddings (서버가 자동으로 로드하므로 API 요청에 포함 불필요)
    # selfIntroductionEmbedding: Vector of selfDescription (Candidate uses this)
    selfIntroductionEmbedding: Optional[List[float]] = None
    # roommateCriteriaEmbedding: Vector of roommateDescription (Seeker uses this)
    roommateCriteriaEmbedding: Optional[List[float]] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": 99,  # Seeker ID (다른 ID 사용)
                "gender": "MALE",
                "name": "홍길동",
                "birthYear": 2002,
                "smoker": False,
                "sleepTime": 11,  # 오후 11시~12시 (11)
                "wakeTime": 7,    # 오전 7시~8시 (7)
                "snoring": False,
                "cleaningCycle": "DAILY",
                "drinkingStyle": "RARELY",
                "bugKiller": False,
                "absentDays": ["SUNDAY"],
                "hobby": "독서"
            }]
        }
    }

    @property
    def age(self) -> int:
        return datetime.now().year - self.birthYear


class UserPreferences(BaseModel):
    """검색 조건"""
    # Note: targetGender, targetAgeRange removed. Matching is relative to MyProfile.

    
    # 중요 체크리스트 (가산점 항목)
    preferNonSmoker: bool = False      # 흡연 안하는 사람
    preferGoodAtBugs: bool = False     # 벌레 잘 잡는 사람
    preferQuietSleeper: bool = False   # 코 안 고는 사람
    
    # Note: Text queries are now handled via UserProfile.roommateDescription embedding
    
    
class MatchRequest(BaseModel):
    myProfile: UserProfile
    preferences: UserPreferences
    candidates: List[UserProfile]
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "myProfile": {
                    "id": 99,
                    "gender": "MALE",
                    "name": "홍길동",
                    "birthYear": 2002,
                    "smoker": False,
                    "sleepTime": 11,  # 오후 11시~12시
                    "wakeTime": 7,    # 오전 7시~8시
                    "snoring": False,
                    "cleaningCycle": "DAILY",
                    "drinkingStyle": "RARELY",
                    "bugKiller": False,
                    "absentDays": ["SUNDAY"],
                    "hobby": "독서"
                },
                "preferences": {
                    "preferNonSmoker": True,
                    "preferGoodAtBugs": True,
                    "preferQuietSleeper": False
                },
                "candidates": [{
                    "id": 1,
                    "gender": "MALE",
                    "name": "후보자1",
                    "birthYear": 2002,
                    "smoker": False,
                    "sleepTime": 12,  # 오전 12시~1시
                    "wakeTime": 8,    # 오전 8시~9시
                    "snoring": False,
                    "cleaningCycle": "WEEKLY",
                    "drinkingStyle": "SOMETIMES",
                    "bugKiller": True,
                    "absentDays": [],
                    "hobby": "게임"
                }]
            }]
        }
    }


class MatchResult(BaseModel):
    userId: int
    name: str
    totalScore: float
    rank: int
    matchDetails: dict


