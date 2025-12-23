import os
import numpy as np
import json
import base64
import asyncio
from typing import List, Dict, Optional
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from .models import RepairAnalysisResult, DuplicateReportInfo, RepairResponse

# ==========================================
# 🔧 Configuration & Mock DB
# ==========================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Must be set in .env
genai.configure(api_key=GOOGLE_API_KEY)

# Mock Database for Reports
REPAIR_REPORTS = [] 
NEXT_REPORT_ID = 1

# Lazy Load Models
_clip_model = None

def get_clip_model():
    global _clip_model
    if _clip_model is None:
        print("Loading CLIP Model...")
        _clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
        print("CLIP Model Loaded.")
    return _clip_model

# ==========================================
# 🧠 AI Analysis (Gemini)
# ==========================================

async def analyze_image_with_gemini(image_bytes: bytes) -> RepairAnalysisResult:
    """
    Gemini 3 Flash to analyze image.
    Enforces Korean output and strict JSON structure.
    """
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": RepairAnalysisResult
        }
    )
    
    pil_img = Image.open(BytesIO(image_bytes))
    
    prompt = """
    당신은 시설 관리 및 안전 점검 전문가 AI입니다. 
    제공된 사진을 분석하여 시설물의 고장 상태를 진단하고 JSON 형식으로 응답하세요.
    
    **중요: 반드시 한국어로 작성하세요.**

    **우선순위 판단 기준 (Priority Logic)**:
    1. **CRITICAL (긴급, 점수 9-10)**: 
       - 거주자의 건강/안전 위협 (예: 하수/변기 역류, 가스 누출, 전선 노출, 현관문/보안장치 파손).
       - 주거 불가능 상태 (단전, 단수).
    2. **HIGH (높음, 점수 7-8)**: 
       - 필수 생활 기능 마비 (예: 난방/에어컨 고장, 냉장고 고장, 싱크대 누수).
    3. **MEDIUM (중간, 점수 4-6)**: 
       - 기능상 불편하나 생활 가능 (예: 방문 파손, 식탁 의자 파손, 전등 1개 나감).
    4. **LOW (낮음, 점수 1-3)**: 
       - 미관상 문제 (예: 벽지 찢어짐, 스크래치).

    **분석 항목**:
    - item: 구체적인 물건 명칭 (한국어).
    - issue: 문제 현상 (한국어).
    - severity: CRITICAL, HIGH, MEDIUM, LOW 중 택1.
    - priority_score: 1~10 사이 정수.
    - reasoning: 왜 이 심각도인지 논리적으로 설명 (한국어).
    - description: 상황 요약 (한국어).
    """
    
    try:
        response = model.generate_content([prompt, pil_img])
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return RepairAnalysisResult(
            item="unknown", issue="unknown", 
            severity="MEDIUM", priority_score=5, reasoning=f"API 호출 오류: {str(e)}", 
            description="AI 분석 중 오류가 발생했습니다."
        )
    
    try:
        data = json.loads(response.text)
        return RepairAnalysisResult(**data)
    except Exception as e:
        print(f"Gemini JSON Parse Error: {e}, Raw: {response.text}")
        return RepairAnalysisResult(
            item="unknown", issue="unknown", 
            severity="MEDIUM", priority_score=5, reasoning="분석 실패", 
            description="이미지 분석에 실패했습니다."
        )

# ==========================================
# 🔍 Duplicate Detection (CLIP)
# ==========================================

async def check_duplicates(query_emb, existing_report_ids: List[int], floor: str, room_number: Optional[str] = None) -> List[DuplicateReportInfo]:
    """
    백엔드에서 위치 필터링한 기존 게시물 ID 목록에 대해
    CLIP 벡터 유사도 비교하여 중복 여부 판단.
    """
    duplicates = []
    
    for report in REPAIR_REPORTS:
        if report['id'] not in existing_report_ids:
            continue
            
        if report.get('embedding') is not None:
            sim = util.pytorch_cos_sim(query_emb, report['embedding'])[0][0].item()
            
            # Threshold: 0.80 (80% 이상 유사 = 중복 의심)
            if sim >= 0.80:
                loc_str = f"{report['floor']}층"
                if report.get('room_number'):
                    loc_str += f" {report['room_number']}호"
                else:
                    loc_str += " (공용)"
                    
                duplicates.append(DuplicateReportInfo(
                    reportId=report['id'],
                    similarity=round(sim, 2),
                    description=report['description'],
                    location=loc_str
                ))
    
    duplicates.sort(key=lambda x: x.similarity, reverse=True)
    return duplicates

async def save_report(analysis: RepairAnalysisResult, query_emb, floor: str, room_number: Optional[str] = None):
    """
    In-memory save for future duplicate checks. 
    """
    global NEXT_REPORT_ID
    
    REPAIR_REPORTS.append({
        "id": NEXT_REPORT_ID,
        "floor": floor,
        "room_number": room_number,
        "description": analysis.description,
        "embedding": query_emb
    })
    NEXT_REPORT_ID += 1

# ==========================================
# 🚀 Main Logic
# ==========================================

from .models import RepairRequest

async def process_repair_request(req: RepairRequest) -> RepairResponse:
    # 1. Read Image from Path
    try:
        with open(req.imagePath, "rb") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Image not found at {req.imagePath}")

    # 2. Calculate CLIP Embedding (신규 이미지 벡터 계산)
    pil_img = Image.open(BytesIO(content))
    clip_model = get_clip_model()
    query_emb = clip_model.encode(pil_img, convert_to_numpy=True)
    
    # 3. Analyze (Gemini) - 병렬 처리
    analysis_task = analyze_image_with_gemini(content)
    
    # 4. Check Duplicates (백엔드가 필터링한 ID 목록과 비교)
    duplicates_task = check_duplicates(
        query_emb, 
        req.existingReportIds,
        req.floor, 
        req.room_number
    )
    
    analysis, duplicates = await asyncio.gather(analysis_task, duplicates_task)
    
    # 5. Save current report
    await save_report(analysis, query_emb, req.floor, req.room_number)
    
    return RepairResponse(
        analysis=analysis,
        duplicates=duplicates,
        is_new=(len(duplicates) == 0)
    )
