from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

def get_embedding(text: str, model_type: str = "passage") -> np.ndarray:
    """
    Get embedding from Upstage Solar API.
    model_type: 'passage' (for storing) or 'query' (for searching)
    """
    if not text:
        return np.array([])
        
    model_name = f"solar-embedding-1-large-{model_type}"
    try:
        response = client.embeddings.create(
            input=text,
            model=model_name
        )
        return np.array(response.data[0].embedding, dtype='float32')
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.array([])
