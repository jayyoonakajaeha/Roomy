
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
    
    images = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
    paths = []
    
    for img_name in images:
        abs_path = os.path.abspath(img_name)
        if not os.path.exists(abs_path):
            print(f"Error: {img_name} not found at {abs_path}")
            continue
            
        print(f"Processing {img_name}...")
        pil_img = Image.open(abs_path)
        
        vec = model.encode(pil_img, convert_to_numpy=True)
        
        npy_path = abs_path.replace(".jpg", ".npy")
        np.save(npy_path, vec)
        paths.append((abs_path, npy_path))
        print(f"Saved vector to {npy_path}")
        
    return paths



def run_verification():
    # No Mocking Gemini. Real API Call will happen.
    
    # Generate Real Vectors
    files = generate_vectors()
    # files[0]=test1, [1]=test2, [2]=test3, [3]=test4
    if len(files) < 4:
        print(f"Not enough images found. Found {len(files)}, need 4.")
        return

    
    # Debug: Print Pairwise Similarities
    from sentence_transformers import util
    import torch
    
    vecs = [np.load(p[1]) for p in files]
    # Convert to tensor for util
    tensors = [torch.tensor(v) for v in vecs]
    
    print("\n[Debug] Pairwise Similarities:")
    print(f"Test1 (Base) vs Test2 (Dup?): {util.pytorch_cos_sim(tensors[0], tensors[1]).item():.4f}")
    print(f"Test1 (Base) vs Test3 (Dup?): {util.pytorch_cos_sim(tensors[0], tensors[2]).item():.4f}")
    if len(files) > 3:
        print(f"Test1 (Base) vs Test4 (New?): {util.pytorch_cos_sim(tensors[0], tensors[3]).item():.4f}")

    # Scenario:
    # 1. Report test1 (New)
    # 2. Report test2 (Duplicate)
    # 3. Report test3 (Duplicate)
    # 4. Report test4 (New)
    
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
    import json
    print("Response JSON:\n", json.dumps(res1.json(), indent=2, ensure_ascii=False))
    
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
    print("Is New:", res2.json()['is_new'])
    print("Duplicates Found:", len(res2.json()['duplicates']))
    
    print("\n--- Test 3: Third Report (test3.jpg) - Should be Duplicate ---")
    req3 = {
        "imagePath": files[2][0],
        "vectorPath": files[2][1],
        "building": building,
        "floor": floor,
        "room_number": "101"
    }
    res3 = client.post("/api/repair/analyze", json=req3)
    print("Is New:", res3.json()['is_new'])
    
    print("\n--- Test 4: Fourth Report (test4.jpg) - Should be NEW (Diff Issue) ---")
    req4 = {
        "imagePath": files[3][0],
        "vectorPath": files[3][1],
        "building": building,
        "floor": floor,
        "room_number": "101"
    }
    res4 = client.post("/api/repair/analyze", json=req4)
    print("Status:", res4.status_code)
    print("Response JSON:\n", json.dumps(res4.json(), indent=2, ensure_ascii=False))
    
    data4 = res4.json()
    print("Is New:", data4['is_new'])
    
    if data4['is_new'] is True:
        print("SUCCESS: Detected as NEW issue.")
    else:
        print("FAIL: Treated as Duplicate.")
        print("Duplicates:", data4['duplicates'])

if __name__ == "__main__":
    run_verification()
