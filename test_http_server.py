"""
Test script to verify API server is working via real HTTP requests
"""
import httpx
import os
import json

# Server URL
BASE_URL = "http://localhost:8001"

def test_matching_api():
    """Test the roommate matching API"""
    print("Testing Roommate Matching API...")
    
    payload = {
        "building": "Dorm A",
        "floor": "3",
        "member_ids": ["student001", "student002"]
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/api/matching/match", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Match Score: {data.get('match_score')}")
            print("SUCCESS ✓")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

def test_repair_api():
    """Test the repair analysis API with real image"""
    print("\nTesting Repair Analysis API...")
    
    # Check if test images exist
    if not os.path.exists("test1.jpg"):
        print("ERROR: test1.jpg not found. Skipping repair API test.")
        return
    
    # For HTTP API, we need to send actual paths that the server can access
    # Since server reads from paths, we use absolute paths
    img_path = os.path.abspath("test1.jpg")
    vec_path = os.path.abspath("test1.npy")
    
    if not os.path.exists(vec_path):
        print(f"ERROR: {vec_path} not found. Please run verify_duplicates_real.py first.")
        return
    
    payload = {
        "imagePath": img_path,
        "vectorPath": vec_path,
        "building": "Dorm Test",
        "floor": "1",
        "room_number": "101"
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/api/repair/analyze", json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("\nResponse JSON:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("\nSUCCESS ✓")
            
            # Verify repair_suggestion is NOT present
            if 'repair_suggestion' in data.get('analysis', {}):
                print("WARNING: repair_suggestion field found (should be removed)")
            else:
                print("VERIFIED: repair_suggestion field successfully removed")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print(f"Testing API server at {BASE_URL}\n")
    print("=" * 60)
    
    test_matching_api()
    test_repair_api()
    
    print("\n" + "=" * 60)
    print("Tests complete!")
