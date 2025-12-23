import os
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
from PIL import Image

client = TestClient(app)

def create_temp_files():
    # 1. Image
    img_path = os.path.abspath("temp_test_image.jpg")
    image = Image.new('RGB', (100, 100), color='red')
    image.save(img_path, 'jpeg')
    
    # 2. Vector
    vec_path = os.path.abspath("temp_test_vector.npy")
    # CLIP vector dim is 512
    vec = np.random.rand(512).astype(np.float32)
    np.save(vec_path, vec)
    
    return img_path, vec_path

def cleanup_files(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

from app.repair.models import RepairAnalysisResult

@patch("app.repair.service.analyze_image_with_gemini")
@patch("app.repair.service.check_duplicates")
def test_repair_analyze(mock_duplicates, mock_gemini):
    # Setup Mocks
    mock_gemini.return_value = RepairAnalysisResult(
        item="toilet",
        issue="clogged",
        severity="CRITICAL",
        priority_score=9,
        reasoning="위생 문제",
        description="변기 역류함."
    )
    
    mock_duplicates.return_value = [
        {
            "reportId": 10,
            "similarity": 0.95,
            "description": "변기 막힘 (이전 신고)",
            "location": "Dorm A 3F"
        }
    ]
    
    # Create Real Files
    img_path, vec_path = create_temp_files()
    
    try:
        # Prepare JSON Request
        data = {
            "imagePath": img_path,
            "vectorPath": vec_path,
            "building": "Dorm A",
            "floor": "3",
            "room_number": "301"
        }
        
        # Request (JSON)
        response = client.post("/api/repair/analyze", json=data)
        
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(response.json())
        
        assert response.status_code == 200
        res_json = response.json()
        
        print("Analysis:", res_json['analysis'])
        print("Duplicates:", res_json['duplicates'])
        
        assert res_json['analysis']['item'] == "toilet"
        assert res_json['analysis']['severity'] == "CRITICAL"
        assert len(res_json['duplicates']) == 1
        
    finally:
        cleanup_files([img_path, vec_path])

if __name__ == "__main__":
    test_repair_analyze()
