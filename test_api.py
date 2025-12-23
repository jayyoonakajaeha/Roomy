from fastapi.testclient import TestClient
from app.main import app
import numpy as np

client = TestClient(app)

def test_matching():
    # 1. Dummy Data Prep
    # 5-dimensional random embeddings
    
    # User A (Seeker)
    seeker_profile = {
        "id": 99,
        "name": "Me",
        "gender": "MALE",
        "birthYear": 2002, 
        "smoker": False,
        "sleepTime": 14, 
        "wakeTime": 7,   
        "snoring": False,
        "cleaningCycle": "DAILY",
        "drinkingStyle": "RARELY",
        "bugKiller": False,
        "absentDays": [],
        "hobby": "Reading",
        "selfDescription": "I am quiet.",
        "roommateDescription": "I want someone quiet.", # Logic uses this for vector search
        "selfIntroductionEmbedding": [0.0, 0.0, 0.0, 0.0, 0.0],
        # Seeker's criteria embedding (Targeting 'Quiet')
        "roommateCriteriaEmbedding": [0.1, 0.2, 0.3, 0.4, 0.5] 
    }
    
    preferences = {
        "targetGender": "MALE",
        "targetAgeRange": [20, 25],
        "preferNonSmoker": True,
        "preferGoodAtBugs": True,
        "queryEmbedding": None # Deprecated/Unused
    }
    
    # Candidates
    candidates = [
        # Match 1: Perfect Match 
        # Matches Seeker's criteria embedding with his selfIntro
        {
            "id": 1, "name": "Perfect Match", "gender": "MALE", "birthYear": 2002,
            "smoker": False, "sleepTime": 14, "wakeTime": 7, 
            "snoring": False, "cleaningCycle": "DAILY", "drinkingStyle": "RARELY",
            "bugKiller": True,
            "selfDescription": "I am a quiet person.",
            "roommateDescription": "Whatever.",
            "selfIntroductionEmbedding": [0.1, 0.2, 0.3, 0.4, 0.5], # Matches Seeker's criteria
            "roommateCriteriaEmbedding": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        # Match 2: Smoker (Pref penalty)
        {
            "id": 2, "name": "Smoker Match", "gender": "MALE", "birthYear": 2002,
            "smoker": True, "sleepTime": 14, "wakeTime": 7,
            "snoring": False, "cleaningCycle": "DAILY", "drinkingStyle": "RARELY",
            "bugKiller": True,
            "selfDescription": "I am quiet but smoke.",
            "selfIntroductionEmbedding": [0.1, 0.2, 0.3, 0.4, 0.5], # Matches criteria vector
            "roommateCriteriaEmbedding": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        # Match 3: Low Text Similarity
        {
            "id": 3, "name": "Dirty Match", "gender": "MALE", "birthYear": 2002,
            "smoker": False, "sleepTime": 14, "wakeTime": 7,
            "snoring": False, "cleaningCycle": "NEVER", "drinkingStyle": "RARELY",
            "bugKiller": True,
            "selfDescription": "I am loud and dirty.",
            "selfIntroductionEmbedding": [0.9, 0.8, 0.7, 0.6, 0.5], # Different vector
            "roommateCriteriaEmbedding": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
    
    payload = {
        "myProfile": seeker_profile,
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
            
        # Assertions
        assert results[0]['userId'] == 1
        assert results[0]['totalScore'] >= 99.0 
        
        # Rank 2: Smoker Match (Score ~85)
        # Rank 3: Dirty Match (Score < 85 because Text Score is low)
        # Dirty Match has Tag Penalty (7.5pts lost) but Text Penalty (Large if vector dissimilar)
        # Vector [0.9...0.5] vs [0.1...0.5] should be low sim.
        
        # Dirty Match (Rank 2) > Smoker Match (Rank 3)
        # Dirty Score: 86.6 (Tag 32.5 + Pref 30 + Text 24.1)
        # Smoker Score: 85.0 (Tag 40 + Pref 15 + Text 30)
        assert results[1]['userId'] == 3 
        assert results[2]['userId'] == 2

if __name__ == "__main__":
    test_matching()
