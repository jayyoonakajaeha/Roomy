import os
import numpy as np
from app.core.embedding import get_embedding

VECTOR_STORAGE_PATH = "storage/vectors"

def ensure_vector_storage():
    if not os.path.exists(VECTOR_STORAGE_PATH):
        os.makedirs(VECTOR_STORAGE_PATH)

def save_user_vectors(user_id: int, self_desc: str, room_desc: str):
    """
    Generate and save embeddings for a user.
    """
    ensure_vector_storage()
    
    # 1. Self Description Embedding (Candidate uses this)
    # Stored as 'passage' type (to be searched against)
    if self_desc:
        self_emb = get_embedding(self_desc, "passage")
        if self_emb.size > 0:
            np.save(os.path.join(VECTOR_STORAGE_PATH, f"{user_id}_self.npy"), self_emb)
            
    # 2. Roommate Description Embedding (Seeker uses this)
    # Stored as 'query' type (to search with) -> Wait, usually query is generated at runtime.
    # But if we want to store it, we should store it as is. 
    # Actually, for FAISS, we need the *query vector* to search against the *passage vectors*.
    # When I search for a roommate, I use my 'roommateDescription' as the query.
    # So I should pre-calculate the query embedding for efficient recurrent searching?
    # Yes, let's store it.
    if room_desc:
        room_emb = get_embedding(room_desc, "query")
        if room_emb.size > 0:
            np.save(os.path.join(VECTOR_STORAGE_PATH, f"{user_id}_criteria.npy"), room_emb)

def load_user_vector(user_id: int, vector_type: str) -> np.ndarray:
    """
    Load vector from storage.
    vector_type: 'self' or 'criteria'
    """
    path = os.path.join(VECTOR_STORAGE_PATH, f"{user_id}_{vector_type}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None
