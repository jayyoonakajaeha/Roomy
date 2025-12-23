import os
import numpy as np
import shutil
from app.users.service import save_user_vectors, load_user_vector

def test_vector_storage():
    # Setup
    test_user_id = 9999
    self_text = "I am a quiet student."
    room_text = "Looking for a clean roommate."
    
    print("1. Saving vectors...")
    save_user_vectors(test_user_id, self_text, room_text)
    
    print("2. Verifying files exist...")
    self_path = f"storage/vectors/{test_user_id}_self.npy"
    room_path = f"storage/vectors/{test_user_id}_criteria.npy"
    
    if os.path.exists(self_path):
        print(f"✅ Self vector file found: {self_path}")
    else:
        print(f"❌ Self vector file missing")
        
    if os.path.exists(room_path):
        print(f"✅ Criteria vector file found: {room_path}")
    else:
        print(f"❌ Criteria vector file missing")

    print("3. Loading and checking dimensions...")
    self_vec = load_user_vector(test_user_id, 'self')
    room_vec = load_user_vector(test_user_id, 'criteria')
    
    if self_vec is not None:
        print(f"✅ Self vector loaded. Shape: {self_vec.shape}")
        # Upstage Solar Large Embedding Dimension is usually 4096
    else:
        print("❌ Failed to load self vector")

    if room_vec is not None:
        print(f"✅ Criteria vector loaded. Shape: {room_vec.shape}")
    else:
        print("❌ Failed to load criteria vector")
        
    # Cleanup
    # if os.path.exists(self_path): os.remove(self_path)
    # if os.path.exists(room_path): os.remove(room_path)
    # print("Cleanup done.")

if __name__ == "__main__":
    test_vector_storage()
