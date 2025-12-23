
import os
import numpy as np
import asyncio
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.repair.service import get_clip_model
from app.repair.models import RepairAnalysisResult
from PIL import Image

# Setup Client
client = TestClient(app)

def generate_vectors():
    print("Loading CLIP Model...")
    model = get_clip_model()
    
    images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    paths = []
    
    for img_name in images:
        abs_path = os.path.abspath(img_name)
        if not os.path.exists(abs_path):
            print(f"Error: {img_name} not found at {abs_path}")
            continue
            
        print(f"Processing {img_name}...")
        pil_img = Image.open(abs_path)
        
        # Generate Vector
        # model.encode returns Tensor or ndarray depending on options.
        # But in service.py we use standard sentence-transformers usage.
        # It typically returns numpy if convert_to_tensor=False (default).
        # Let's check service.py usage: convert_to_tensor=True. 
        # But here we want to save it as .npy.
        
        vec = model.encode(pil_img, convert_to_numpy=True)
        # Verify shape
        # print(vec.shape) # Should be (512,)
        
        npy_path = abs_path.replace(".jpg", ".npy")
        np.save(npy_path, vec)
        paths.append((abs_path, npy_path))
        print(f"Saved vector to {npy_path}")
        
    return paths

@patch("app.repair.service.analyze_image_with_gemini")
def run_verification(mock_gemini):
    # Mock Gemini Logic
    mock_gemini.return_value = RepairAnalysisResult(
        item="toilet",
        issue="clogged",
        severity="CRITICAL",
        priority_score=9,
        reasoning="Test",
        repair_suggestion="Test",
        description="Test Case"
    )
    
    # Generate Real Vectors
    files = generate_vectors()
    if len(files) < 3:
        print("Not enough images found. Need test1, test2, test3.")
        return

    
    # Debug: Print Pairwise Similarities
    from sentence_transformers import util
    import torch
    
    vecs = [np.load(p[1]) for p in files]
    # Convert to tensor for util
    tensors = [torch.tensor(v) for v in vecs]
    
    print("\n[Debug] Pairwise Similarities:")
    print(f"Test1 vs Test2: {util.pytorch_cos_sim(tensors[0], tensors[1]).item():.4f}")
    if len(files) > 2:
        print(f"Test1 vs Test3: {util.pytorch_cos_sim(tensors[0], tensors[2]).item():.4f}")
        print(f"Test2 vs Test3: {util.pytorch_cos_sim(tensors[1], tensors[2]).item():.4f}")

    # Scenario:
    # 1. Report test1 (New)
    # 2. Report test2 (Duplicate)
    # 3. Report test3 (Duplicate)
    
    building = "Dorm Test"
    floor = "1"
    
    print("\n--- Test 1: First Report (test1.jpg) ---")
    req1 = {
        "imagePath": files[0][0],
        "vectorPath": files[0][1],
        "building": building,
        "floor": floor,
        "room_number": "101"
    }
    res1 = client.post("/api/repair/analyze", json=req1)
    print("Status:", res1.status_code)
    print("Is New:", res1.json()['is_new'])
    assert res1.json()['is_new'] == True
    
    print("\n--- Test 2: Second Report (test2.jpg) - Should be Duplicate ---")
    req2 = {
        "imagePath": files[1][0],
        "vectorPath": files[1][1],
        "building": building,
        "floor": floor,
        "room_number": "101"
    }
    res2 = client.post("/api/repair/analyze", json=req2)
    print("Status:", res2.status_code)
    data2 = res2.json()
    print("Is New:", data2['is_new'])
    print("Duplicates Found:", len(data2['duplicates']))
    if len(data2['duplicates']) > 0:
        print("Similarity:", data2['duplicates'][0]['similarity'])
    
    # Assert
    if data2['is_new'] is False:
        print("SUCCESS: Detected as duplicate.")
    else:
        print("FAIL: Treated as new report. Similarity might be too low?")

    print("\n--- Test 3: Third Report (test3.jpg) - Should be Duplicate ---")
    req3 = {
        "imagePath": files[2][0],
        "vectorPath": files[2][1],
        "building": building,
        "floor": floor,
        "room_number": "101"
    }
    res3 = client.post("/api/repair/analyze", json=req3)
    print("Status:", res3.status_code)
    data3 = res3.json()
    print("Is New:", data3['is_new'])
    if data3['is_new'] is False:
        print("SUCCESS: Detected as duplicate.")
    else:
         print("FAIL: Treated as new report.")

if __name__ == "__main__":
    run_verification()
