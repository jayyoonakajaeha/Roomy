import httpx
import os

def test_vector_generation_api():
    url = "http://127.0.0.1:8002/api/users/vector"
    payload = {
        "userId": 777,
        "selfDescription": "I am a test user for vector API.",
        "roommateDescription": "I want a test roommate."
    }
    
    print(f"Sending request to {url}...")
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            # Verify file creation
            self_path = "storage/vectors/777_self.npy"
            criteria_path = "storage/vectors/777_criteria.npy"
            
            if os.path.exists(self_path) and os.path.exists(criteria_path):
                print("✅ Vector files created successfully!")
            else:
                print("❌ Vector files missing!")
        else:
            print("❌ API failed!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_vector_generation_api()
