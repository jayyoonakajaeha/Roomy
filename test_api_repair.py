from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock
from io import BytesIO
from PIL import Image

client = TestClient(app)

def create_dummy_image():
    file = BytesIO()
    image = Image.new('RGB', (100, 100), color='red')
    image.save(file, 'jpeg')
    file.seek(0)
    return file

from app.repair.models import RepairAnalysisResult

@patch("app.repair.service.analyze_image_with_gemini")
@patch("app.repair.service.check_duplicates")
def test_repair_analyze(mock_duplicates, mock_gemini):
    # 1. Setup Mocks
    mock_gemini.return_value = RepairAnalysisResult(
        category="plumbing",
        item="toilet",
        issue="clogged",
        severity="CRITICAL",
        priority_score=9,
        reasoning="위생 문제",
        repair_suggestion="뚫어뻥",
        description="변기 역류함."
    )
    
    # Simulate one duplicate found
    mock_duplicates.return_value = [
        {
            "reportId": 10,
            "similarity": 0.95,
            "description": "변기 막힘 (이전 신고)",
            "location": "Dorm A 3F"
        }
    ]
    
    # 2. Prepare Request
    img_file = create_dummy_image()
    files = {"file": ("test.jpg", img_file, "image/jpeg")}
    data = {"building": "Dorm A", "floor": "3"}
    
    # 3. Request
    response = client.post("/api/repair/analyze", files=files, data=data)
    
    # 4. Assertions
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    res_json = response.json()
    
    print("Analysis:", res_json['analysis'])
    print("Duplicates:", res_json['duplicates'])
    
    assert res_json['analysis']['category'] == "plumbing"
    assert res_json['analysis']['severity'] == "CRITICAL"
    assert len(res_json['duplicates']) == 1
    assert res_json['duplicates'][0]['similarity'] == 0.95
    assert res_json['is_new'] is False # Because duplicate exists

if __name__ == "__main__":
    test_repair_analyze()
