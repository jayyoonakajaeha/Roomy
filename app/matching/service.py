import numpy as np
import faiss
from typing import List
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
    # Target: Candidates' selfIntroductionEmbedding
    # Query: Seeker's roommateCriteriaEmbedding
    
    valid_candidates_with_emb = [c for c in candidates if c.selfIntroductionEmbedding is not None and len(c.selfIntroductionEmbedding) > 0]
    text_scores_map = {} 
    
    # Seeker must have roommateCriteriaEmbedding to search
    if valid_candidates_with_emb and seeker.roommateCriteriaEmbedding:
        d = len(seeker.roommateCriteriaEmbedding)
        index = faiss.IndexFlatIP(d) 
        
        # Candidate Matrix (Self Description Vectors)
        candidate_matrix = np.array([c.selfIntroductionEmbedding for c in valid_candidates_with_emb], dtype='float32')
        faiss.normalize_L2(candidate_matrix)
        index.add(candidate_matrix)
        
        # Query Vector (Seeker's Roommate Criteria)
        query_vec = np.array([seeker.roommateCriteriaEmbedding], dtype='float32')
        faiss.normalize_L2(query_vec)
        
        k = len(valid_candidates_with_emb)
        D, I = index.search(query_vec, k)
        
        distances = D[0]
        indices = I[0]
        
        for i, idx in enumerate(indices):
            if idx == -1: continue
            c_user = valid_candidates_with_emb[idx]
            # Cosine Sim (-1~1) -> 0~1 Scale
            sim = max(0.0, float(distances[i]))
            text_scores_map[c_user.id] = sim # 0.0 ~ 1.0

    # 2. Scoring Loop
    results = []
    
    # Weights Configuration (Total 100)
    W_TAG = 40.0
    W_PREF = 30.0
    W_TEXT = 30.0
    
    for cand in candidates:
        if cand.id == seeker.id: continue
        
        # Hard Filter (성별)
        if prefs.targetGender != cand.gender:
            continue

        # --- A. Tag Score (40점 만점) ---
        # 1. Age (10점)
        cand_age = cand.age
        min_age, max_age = prefs.targetAgeRange
        if min_age <= cand_age <= max_age:
            age_p = 1.0
        else:
            diff = min(abs(cand_age - min_age), abs(cand_age - max_age))
            age_p = max(0.0, 1.0 - (diff * 0.1)) # 1살 차이당 10% 감점
        
        # 2. Time (15점) -> Wake(7.5) + Sleep(7.5)
        # Wake range: 5~11 (max diff 6)
        wake_p = get_scale_diff_score(seeker.wakeTime, cand.wakeTime, 6)
        # Sleep range: 8~14 (max diff 6)
        sleep_p = get_scale_diff_score(seeker.sleepTime, cand.sleepTime, 6)
        time_p = (wake_p + sleep_p) / 2.0
        
        # 3. Habits (15점) -> Cleaning(7.5) + Drinking(7.5)
        # Cleaning range: 0~4 (max diff 4)
        clean_p = get_scale_diff_score(seeker.cleaningCycle.to_score(), cand.cleaningCycle.to_score(), 4)
        # Drinking range: 0~2 (max diff 2)
        drink_p = get_scale_diff_score(seeker.drinkingStyle.to_score(), cand.drinkingStyle.to_score(), 2)
        habit_p = (clean_p + drink_p) / 2.0
        
        tag_score = (age_p * 10.0) + (time_p * 15.0) + (habit_p * 15.0)

        # --- B. Preference Score (30점 만점) ---
        active_prefs = []
        if prefs.preferNonSmoker: active_prefs.append(lambda u: not u.smoker)
        if prefs.preferGoodAtBugs: active_prefs.append(lambda u: u.bugKiller)
        if prefs.preferQuietSleeper: active_prefs.append(lambda u: not u.snoring)
        if prefs.preferNonDrinker: active_prefs.append(lambda u: u.drinkingStyle == DrinkingStyle.RARELY)
        
        pref_ratio = 0.0
        if len(active_prefs) > 0:
            matched_cnt = sum(1 for check in active_prefs if check(cand))
            pref_ratio = matched_cnt / len(active_prefs)
            
        pref_score = pref_ratio * W_PREF
        
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
