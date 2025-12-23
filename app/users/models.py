from pydantic import BaseModel
from typing import Optional

class VectorGenerationRequest(BaseModel):
    userId: int
    selfDescription: Optional[str] = None
    roommateDescription: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "userId": 1,
                "selfDescription": "저는 조용하고 깔끔한 성격입니다.",
                "roommateDescription": "비흡연자이고 조용한 사람을 원합니다."
            }
        }
