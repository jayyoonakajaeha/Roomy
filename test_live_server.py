import httpx
import json
from app.users.service import save_user_vectors

def test_live_server():
    print("1. Preparing Vectors on Disk...")
    # Seeker (99): Wants "Clean, Non-Smoker"
    save_user_vectors(99, None, "clean non-smoker")
    
    # Candidate (1): Is "Clean, Non-Smoker"
    save_user_vectors(1, "I am clean and non-smoker", None)
    
    print("2. Checking Server Health...")
    try:
        health_res = httpx.get("http://127.0.0.1:8001/")
        print(f"Health Check: {health_res.status_code}")
        print(health_res.text)
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return

    print("3. Sending Match Request to http://127.0.0.1:8001/api/matching/match ...")
    
    url = "http://127.0.0.1:8001/api/matching/match"
    
    payload = {
        "myProfile": {
            "id": 99,
            "gender": "MALE",
            "birthYear": 2002,
            "name": "LiveTester",
            "smoker": False,
            "sleepTime": 14,
            "wakeTime": 7,
            "cleaningCycle": "DAILY",
            "drinkingStyle": "RARELY",
            "snoring": False,
            "bugKiller": False,
            "intro": "Living Test",
            "hobby": "Testing"
        },
        "preferences": {
            "preferNonSmoker": True,
            "preferGoodAtBugs": True,
            "preferQuietSleeper": True,
            "preferNonDrinker": True
        },
        "candidates": [
            {
                "id": 1,
                "name": "LiveCandidate",
                "gender": "MALE",
                "birthYear": 2002,
                "smoker": False,
                "sleepTime": 14,
                "wakeTime": 7,
                "cleaningCycle": "DAILY",
                "drinkingStyle": "RARELY",
                "snoring": False,
                "bugKiller": False,
                "intro": "I am live",
                "hobby": "Being Live"
            }
        ]
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response Body:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print("Error Response:")
            print(response.content)
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_live_server()
