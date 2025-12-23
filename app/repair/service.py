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

async def save_report_files(new_id: int, temp_image_path: str, query_emb, floor: str, room_number: Optional[str] = None, description: str = ""):
    """
    중복이 아닌 경우: 임시 이미지를 영구 저장소로 이동, 임베딩 저장.
    - 임베딩 벡터: storage/repair_vectors/{new_id}.npy
    - 이미지 이동: storage/temp/pending.jpg → storage/repair_images/{new_id}.jpg
    """
    import shutil
    
    # 디렉토리 생성
    os.makedirs("storage/repair_vectors", exist_ok=True)
    os.makedirs("storage/repair_images", exist_ok=True)
    
    # 1. 임베딩 저장
    vector_path = f"storage/repair_vectors/{new_id}.npy"
    np.save(vector_path, query_emb)
    
    # 2. 임시 이미지 → 영구 저장소로 이동
    _, ext = os.path.splitext(temp_image_path)
    new_image_path = f"storage/repair_images/{new_id}{ext}"
    shutil.move(temp_image_path, new_image_path)
    
    # 3. In-memory 저장 (테스트용)
    REPAIR_REPORTS.append({
        "id": new_id,
        "floor": floor,
        "room_number": room_number,
        "description": description,
        "embedding": query_emb
    })
    
    return new_image_path

def delete_temp_image(temp_image_path: str):
    """중복 신고인 경우 임시 이미지 삭제"""
    try:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    except Exception as e:
        print(f"Failed to delete temp image: {e}")

# ==========================================
# 🚀 Main Logic
# ==========================================

from .models import RepairRequest

# 고정 임시 이미지 경로
TEMP_IMAGE_PATH = "storage/temp/pending.jpg"

async def process_repair_request(req: RepairRequest) -> RepairResponse:
    # 1. Read Image from Fixed Path
    try:
        with open(TEMP_IMAGE_PATH, "rb") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Image not found at {TEMP_IMAGE_PATH}")

    # 2. Calculate CLIP Embedding (신규 이미지 벡터 계산)
    pil_img = Image.open(BytesIO(content))
    clip_model = get_clip_model()
    query_emb = clip_model.encode(pil_img, convert_to_numpy=True)
    
    # 3. Check Duplicates FIRST (중복이면 Gemini 호출 안함 = 토큰 절약)
    duplicates = await check_duplicates(
        query_emb, 
        req.existingReportIds,
        req.floor, 
        req.room_number
    )
    
    is_new = len(duplicates) == 0
    
    if is_new:
        # 4. 신규일 때만 Gemini 분석
        analysis = await analyze_image_with_gemini(content)
        new_id = req.totalReportCount + 1
        
        # 5. 파일 저장 (description 포함)
        await save_report_files(new_id, TEMP_IMAGE_PATH, query_emb, req.floor, req.room_number, analysis.description)
    else:
        # 중복: Gemini 스킵, 임시 파일 삭제
        analysis = None
        new_id = None
        delete_temp_image(TEMP_IMAGE_PATH)
    
    return RepairResponse(
        analysis=analysis,
        duplicates=duplicates,
        is_new=is_new,
        newReportId=new_id
    )

