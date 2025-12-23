from fastapi import APIRouter, HTTPException
from app.users.models import VectorGenerationRequest
from app.users.service import save_user_vectors

router = APIRouter()

@router.post("/vector", summary="Generate and save user vectors")
async def generate_user_vectors(request: VectorGenerationRequest):
    """
    Generate embedding vectors for user descriptions and save them to storage.
    These vectors are used for roommate matching.
    """
    try:
        # Generate and save vectors
        # If descriptions are None, save_user_vectors handles it gracefully (skips saving)
        save_user_vectors(
            user_id=request.userId,
            self_desc=request.selfDescription,
            room_desc=request.roommateDescription
        )
        
        return {
            "status": "ok",
            "message": f"Vectors saved for user {request.userId}",
            "details": {
                "self_vector": "Saved" if request.selfDescription else "Skipped",
                "criteria_vector": "Saved" if request.roommateDescription else "Skipped"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
