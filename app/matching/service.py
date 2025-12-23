import numpy as np
import faiss
from typing import List
from app.users.service import load_user_vector
from .models import MatchRequest, MatchResult, DrinkingStyle

# ==========================================
# 📏 Helper Functions
# ==========================================

def get_scale_diff_score(val1: int, val2: int, max_diff_range: int) -> float:
    """단순 선형 차이 점수 (0.0 ~ 1.0)"""
    diff = abs(val1 - val2)
    # 차이가 max_diff_range보다 크면 0점 처리
    normalized_diff = min(diff, max_diff_range) / max_diff_range
    return max(0.0, 1.0 - normalized_diff)

# ==========================================
# 🧠 Matching Logic
# ==========================================

def calculate_hybrid_match(request: MatchRequest) -> List[MatchResult]:
    seeker = request.myProfile
    prefs = request.preferences
    candidates = request.candidates
    
    # 1. FAISS Vector Search 준비
    # Load embeddings directly from storage
    
    # 1-1. Seeker Criteria Embedding
    seeker_vec = None
    if seeker.roommateCriteriaEmbedding:
        # If provided in request (fallback/debug), use it
        seeker_vec = np.array([seeker.roommateCriteriaEmbedding], dtype='float32')
    else:
        # Load from disk
        loaded = load_user_vector(seeker.id, 'criteria') # {id}_criteria.npy
        if loaded is not None:
             seeker_vec = np.array([loaded], dtype='float32')
             
    # 1-2. Candidates Self Embeddings
    valid_candidates_with_emb = []
    candidate_vectors = []
    
    for c in candidates:
        if c.selfIntroductionEmbedding:
             # Provided in request
             valid_candidates_with_emb.append(c)
             candidate_vectors.append(c.selfIntroductionEmbedding)
        else:
             # Load from disk
             loaded = load_user_vector(c.id, 'self') # {id}_self.npy
             if loaded is not None:
                 valid_candidates_with_emb.append(c)
                 candidate_vectors.append(loaded)
    
    text_scores_map = {} 
    
    if seeker_vec is not None and len(candidate_vectors) > 0:
        d = seeker_vec.shape[1]
        index = faiss.IndexFlatIP(d) 
        
        # Candidate Matrix
        candidate_matrix = np.array(candidate_vectors, dtype='float32')
        faiss.normalize_L2(candidate_matrix)
        index.add(candidate_matrix)
        
        # Query Vector
        faiss.normalize_L2(seeker_vec)
        
        k = len(candidate_vectors)
        D, I = index.search(seeker_vec, k)
        
        distances = D[0]
        indices = I[0]
        
        for i, idx in enumerate(indices):
            if idx == -1: continue
            c_user = valid_candidates_with_emb[idx]
            sim = max(0.0, float(distances[i]))
            text_scores_map[c_user.id] = sim

    # 2. Scoring Loop
    results = []
    
    # Weights Configuration (Total 100)
    W_TAG = 40.0
    W_PREF = 30.0
    W_TEXT = 30.0
    
    for cand in candidates:
        if cand.id == seeker.id: continue
        
        # Hard Filter (Gender) removed per user request

        # --- A. Tag Score (40점 만점) ---
        # 1. Age (10점 배점)
        # 0살 차이 10점, 1살 차이 9점 ... 10살 이상 0점
        cand_age = cand.age
        seeker_age = seeker.age
        age_diff = abs(seeker_age - cand_age)
        
        # 100점 만점 기준 점수: 100 - (차이 * 10)
        age_base_score = max(0, 100 - (age_diff * 10))
        # 40점 중 5점 비중으로 반영 (0.05 곱하기)
        age_p = age_base_score * 0.05
        
        # 2. Time (15점) -> Wake(7.5) + Sleep(7.5)
        # Wake range: 5~11 (max diff 6)
        wake_p = get_scale_diff_score(seeker.wakeTime, cand.wakeTime, 6)
        # Sleep range: 8~14 (max diff 6)
        sleep_p = get_scale_diff_score(seeker.sleepTime, cand.sleepTime, 6)
        # Time Total: 20점 (기상 10 + 취침 10)
        time_p = (wake_p + sleep_p) / 2.0 * 20.0
        
        # 3. Habits (15점) -> Cleaning(7.5) + Drinking(7.5)
        # Cleaning range: 0~4 (max diff 4)
        clean_p = get_scale_diff_score(seeker.cleaningCycle.to_score(), cand.cleaningCycle.to_score(), 4)
        # Drinking range: 0~2 (max diff 2)
        drink_p = get_scale_diff_score(seeker.drinkingStyle.to_score(), cand.drinkingStyle.to_score(), 2)
        habit_p = (clean_p + drink_p) / 2.0 * 15.0
        
        tag_score = age_p + time_p + habit_p

        # --- B. Preference Score (30점 만점) ---
        active_prefs = []
        if prefs.preferNonSmoker: active_prefs.append(lambda u: not u.smoker)
        if prefs.preferGoodAtBugs: active_prefs.append(lambda u: u.bugKiller)
        if prefs.preferQuietSleeper: active_prefs.append(lambda u: not u.snoring)
        if prefs.preferHeavySleeper: active_prefs.append(lambda u: u.heavySleeper)
        if prefs.preferGoodAtAlarm: active_prefs.append(lambda u: u.goodAtAlarm)
        
        if len(active_prefs) == 0:
            # 선호 조건이 없으면 감점 없음 (만점)
            pref_score = W_PREF
        else:
            matched_cnt = sum(1 for check in active_prefs if check(cand))
            # 만족 비율만큼 점수 부여 (== 불일치 비율만큼 감점)
            pref_score = (matched_cnt / len(active_prefs)) * W_PREF
        
        # --- C. Text Score (30점 만점) ---
        txt_sim = text_scores_map.get(cand.id, 0.0)
        text_score = txt_sim * W_TEXT
        
        # Final Sum
        total_score = tag_score + pref_score + text_score
        
        # 결과 저장
        results.append(MatchResult(
            userId=cand.id,
            name=cand.name,
            totalScore=round(total_score, 1),
            rank=0,
            matchDetails={
                "tagScore": round(tag_score, 1),
                "prefScore": round(pref_score, 1),
                "textScore": round(text_score, 1),
                "age": cand.age
            }
        ))
        
    results.sort(key=lambda x: x.totalScore, reverse=True)
    
    final_results = []
    for i, res in enumerate(results[:20]):
        res.rank = i + 1
        final_results.append(res)
        
    return final_results
