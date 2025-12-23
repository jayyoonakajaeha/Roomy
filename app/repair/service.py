import os
import base64
from openai import OpenAI
from fastapi import UploadFile, HTTPException

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"), # or OPENAI_API_KEY
    base_url="https://api.upstage.ai/v1/solar"
)

async def analyze_repair_image(file: UploadFile) -> dict:
    """
    이미지를 받아 AI(Vision Model)로 분석하여 고장 내용을 추출함.
    """
    try:
        # 1. Read Image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # 2. Call AI
        # Note: Upstage Solar API might not support standard GPT-4o Vision format yet.
        # If this fails, user needs to switch to OpenAI Key or use Upstage's specific OCR/Layout API.
        # For this prototype, we use the standard Chat Completion with Image structure.
        
        response = client.chat.completions.create(
            model="solar-1-mini-chat", # or gpt-4o if using OpenAI
            messages=[
                {
                    "role": "system",
                    "content": "You are a facility maintenance AI. Analyze the image and extract: category (plumbing, electric, furniture, etc.), item (faucet, light, etc.), issue, severity, and a short description. Return valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this repair issue."},
                        # Upstage Solar currently supports text-only via this endpoint usually.
                        # If using OpenAI proper:
                        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            # Upstage specific param for JSON mode if available?
            # response_format={"type": "json_object"} 
        )
        
        # Mock Response (since we can't guarantee Vision support on the current Key)
        # In a real scenario, we parse response.choices[0].message.content
        return {
            "category": "analyzed_category",
            "item": "analyzed_item",
            "issue": "analyzed_issue",
            "severity": "medium",
            "description": "AI analysis result placeholder (Vision API required)"
        }
        
    except Exception as e:
        print(f"AI Analysis Error: {str(e)}")
        # Fallback for demo
        return {
            "error": str(e),
            "category": "unknown",
            "description": "Failed to analyze image with current API configuration."
        }
