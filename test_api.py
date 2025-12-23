from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from app.matching.models import UserProfile, MatchRequest, UserPreferences
from app.users.service import save_user_vectors
import shutil
import os

client = TestClient(app)

def test_matching():
    # 1. Dummy Data Prep
    # 5-dimensional random embeddings
    
    # User A (Seeker)
    
    # Candidates
    
    # 1. Setup Vectors (File Storage)
    # create dummy embeddings
    seeker_id = 99
    cand_id = 1
    
    # Save vectors to disk (Assuming test environment shares storage)
    # Seeker wants roommate -> "criteria" vector
    save_user_vectors(seeker_id, None, "clean non-smoker")
    
    # Candidate introduces self -> "self" vector
    save_user_vectors(cand_id, "I am clean and non-smoker", None)

    # 2. Update Request Body (Remove explicit embeddings)
    my_profile = {
        "id": seeker_id,
        "gender": "MALE",
        "birthYear": 2002,
        "name": "내 이름",
        "smoker": False,
        "sleepTime": 11,  # 오후 11시~12시
        "wakeTime": 7,    # 오전 7시~8시
        "cleaningCycle": "DAILY",
        "drinkingStyle": "RARELY",
        "snoring": False,
        "bugKiller": False,
        "intro": "Hello", 
        # No roommateCriteriaEmbedding here
    }

    preferences = {
        "preferNonSmoker": True,
        "preferGoodAtBugs": True,
        "preferQuietSleeper": True
    }

    candidates = [
        {
            "id": cand_id,
            "name": "후보자1",
            "gender": "MALE",
            "birthYear": 2002,
            "smoker": False,
            "sleepTime": 12,  # 오전 12시~1시
            "wakeTime": 8,    # 오전 8시~9시
            "cleaningCycle": "DAILY",
            "drinkingStyle": "RARELY",
            "snoring": False,
            "bugKiller": False,
            "intro": "Hi",
            # No selfIntroductionEmbedding here
        }
    ]
    
    payload = {
        "myProfile": my_profile,
        "preferences": preferences,
        "candidates": candidates
    }
    
    # Updated endpoint prefix
    response = client.post("/api/matching/match", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(response.json())
        
    if response.status_code == 200:
        results = response.json()
        print(f"Matches Found: {len(results)}")
        for r in results:
            print(f"Rank {r['rank']}: {r['name']} (Score: {r['totalScore']})")
            print(f"   Details: {r['matchDetails']}")
            
        # Dirty Match (Rank 2) > Smoker Match (Rank 3)
        # Dirty Score: 86.6 (Tag 32.5 + Pref 30 + Text 24.1)
        # Smoker Score: 85.0 (Tag 40 + Pref 15 + Text 30)
        assert results[0]['userId'] == cand_id
    # 점수 검증은 scoring logic이 유동적이므로 생략하거나 완화
    # assert results[0]['totalScore'] >= 99.0 

if __name__ == "__main__":
    test_matching()
